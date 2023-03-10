import random, logging, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler

from processors.utils_sc import convert_examples_to_features_sc
from processors.pawsx import PawsxProcessor
from transformers import BertTokenizer

def DataloaderSC(lang_list,
                model_name_or_path,
                model_type,
                mode_list,
                data_dir,
                max_seq_length,
                batch_size, small_train=None):
    lang2id = None # if model_type != 'xlm'
    if 'pawsx' in data_dir.split('/')[-1]:
        processor = PawsxProcessor()
    else:
        raise('no support')
    output_mode = "classification"
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
    
    dataloader = {}
    iter_dataloader = {}
    
    for lang in lang_list:
        dataloader[lang] = {}
        iter_dataloader[lang] = {}
        for mode in mode_list:
            if mode == 'train' and small_train is not None:
                cached_features_file = os.path.join(data_dir, "cached_feature_{}_{}_{}_{}_{}".format(mode, lang,
                                            list(filter(None, model_name_or_path.split("/"))).pop(),
                                            str(max_seq_length), small_train))
            elif mode in ['moml_train', 'moml_val']:
                cached_features_file = os.path.join(data_dir, "cached_feature_train_{}_{}_{}".format(lang,
                                        list(filter(None, model_name_or_path.split("/"))).pop(),
                                        str(max_seq_length)))
            else:
                cached_features_file = os.path.join(data_dir, "cached_feature_{}_{}_{}_{}".format(mode, lang,
                                            list(filter(None, model_name_or_path.split("/"))).pop(),
                                            str(max_seq_length)))
            try:
                features_lg = torch.load(cached_features_file)
                print("Loading features from cached file {}".format(cached_features_file))
            except:
                print("Creating features from dataset file at {} in language {} and mode {}".format(cached_features_file, lang, mode))
                examples = processor.get_examples(data_dir, language=lang, split=mode)
                features_lg = convert_examples_to_features_sc(
                                            examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=False,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0,
                                            lang2id=lang2id,
                                        )
                if mode == 'train' and small_train is not None:
                    random.shuffle(features_lg)
                    features_lg = features_lg[:small_train]
                torch.save(features_lg, cached_features_file)
                
            if mode in ['moml_train', 'moml_val']:
                data_len = len(features_lg)
                train_features = list(np.random.choice(np.array(features_lg), size=int(data_len*0.8), replace=False))
                val_features = list(set(features_lg).difference(set(train_features)))
                features_lg = train_features if mode == 'moml_train' else val_features
       
            all_input_ids = torch.tensor([f.input_ids for f in features_lg], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features_lg], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features_lg], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features_lg], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

            sampler = RandomSampler(dataset) if mode in ['train', 'moml_train', 'moml_val'] else SequentialSampler(dataset)
            drop_last = True if mode in ['train', 'moml_train', 'moml_val'] else False
            dataloader[lang][mode] = DataLoader(dataset, 
                                                sampler=sampler, 
                                                batch_size=batch_size, 
                                                num_workers=2, 
                                                pin_memory=True,
                                                drop_last=drop_last)
            iter_dataloader[lang][mode] = iter(dataloader[lang][mode])
    return dataloader, iter_dataloader, label_list