# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PAWS-X utils (dataset loading and evaluation) """


import logging
import os

from transformers import DataProcessor
from .utils_sc import InputExample


logger = logging.getLogger(__name__)


class PawsxProcessor(DataProcessor):
  """Processor for the PAWS-X dataset."""

  def __init__(self):
    pass

  def get_examples(self, data_dir, language='en', split='train'):
    """See base class."""
    examples = []
    for lg in language.split(','):
      lines = self._read_tsv(os.path.join(data_dir, "{}-{}.tsv".format(split, lg)))
      
      for (i, line) in enumerate(lines):
        if i == 0:
          continue
        guid = "%s-%s-%s" % (split, lg, i)
        text_a = line[0]
        text_b = line[1]
#         if split == 'test' and len(line) != 3:
#           label = "0"
#         else:
        label = str(line[2].strip())
        assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
    return examples

#   def get_translate_examples(self, data_dir, language='en', split='train'):
#     """See base class."""
#     languages = language.split(',')
#     examples = []
#     for language in languages:
#       if split == 'train':
#         file_path = os.path.join(data_dir, "translated/en-{}-translated.tsv".format(language))
#       else:
#         file_path = os.path.join(data_dir, "translated/test-{}-en-translated.tsv".format(language))
#       logger.info("reading from " + file_path)
#       lines = self._read_tsv(file_path)
#       for (i, line) in enumerate(lines):
#         guid = "%s-%s-%s" % (split, language, i)
#         text_a = line[0]
#         text_b = line[1]
#         label = str(line[2].strip())
#         assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
#         examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
#     return examples

#   def get_train_examples(self, data_dir, language='en'):
#     """See base class."""
#     return self.get_examples(data_dir, language, split='train')

#   def get_translate_train_examples(self, data_dir, language='en'):
#     """See base class."""
#     return self.get_translate_examples(data_dir, language, split='train')

#   def get_translate_test_examples(self, data_dir, language='en'):
#     """See base class."""
#     return self.get_translate_examples(data_dir, language, split='test')

#   def get_test_examples(self, data_dir, language='en'):
#     """See base class."""
#     return self.get_examples(data_dir, language, split='test')

#   def get_dev_examples(self, data_dir, language='en'):
#     """See base class."""
#     return self.get_examples(data_dir, language, split='dev')

  def get_labels(self):
    """See base class."""
    return ["0", "1"]


pawsx_processors = {
  "pawsx": PawsxProcessor,
}

pawsx_output_modes = {
  "pawsx": "classification",
}

pawsx_tasks_num_labels = {
  "pawsx": 2,
}
