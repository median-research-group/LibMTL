import torch
import torch.nn as nn
import torch.nn.functional as F

from LibMTL.architecture.abstract_arch import AbsArchitecture

class AOEMTL(AbsArchitecture):
    r"""Autonomy of Experts (AoE-CGC variant).
    Shared + task-specific experts fused via L2-norm weighting.
    """

    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(AOEMTL, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        if self.multi_input:
            raise ValueError("AoE-MTL does not support multi_input=True")

        # Parse expert configuration
        expert_counts = self.kwargs["num_experts"]
        if len(expert_counts) == 1:
            self.num_shared = expert_counts[0]
            self.num_specific = 0
        elif len(expert_counts) == 2:
            self.num_shared, self.num_specific = expert_counts
        else:
            raise ValueError(f"num_experts must have length 1 or 2, got {len(expert_counts)}")

        self.experts_shared = nn.ModuleList([encoder_class() for _ in range(self.num_shared)])

        self.experts_specific = nn.ModuleDict({
            t: nn.ModuleList([encoder_class() for _ in range(self.num_specific)]) for t in self.task_name
        })

        print(f"[AoE-MTL] Initialized with {self.num_shared} shared + {self.num_specific} per-task experts (no gates).")

    def forward(self, inputs, task_name=None):
        out = {}
        shared_outs = [e(inputs) for e in self.experts_shared]
        shared_stack = torch.stack(shared_outs, dim=1)

        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue

            if self.num_specific > 0:
                spec_outs = [e(inputs) for e in self.experts_specific[task]]
                spec_stack = torch.stack(spec_outs, dim=1)
                all_experts = torch.cat([shared_stack, spec_stack], dim=1)
            else:
                all_experts = shared_stack

            l2_norms = torch.norm(all_experts.flatten(2), dim=2)  
            weights = l2_norms / (l2_norms.sum(dim=1, keepdim=True) + 1e-8)
            weights_exp = weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)

            fused = (all_experts * weights_exp).sum(dim=1)  

            feat = self._prepare_rep(fused, task, same_rep=False)
            out[task] = self.decoders[task](feat)

        return out

    def get_share_params(self):
        return self.experts_shared.parameters()

    def zero_grad_share_params(self):
        self.experts_shared.zero_grad(set_to_none=False)
