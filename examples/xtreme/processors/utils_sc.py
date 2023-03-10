import copy
import csv
import json
import logging
from transformers import XLMTokenizer

logger = logging.getLogger(__name__)


class InputExample(object):
  """
  A single training/test example for simple sequence classification.
  Args:
    guid: Unique id for the example.
    text_a: string. The untokenized text of the first sequence. For single
    sequence tasks, only this sequence must be specified.
    text_b: (Optional) string. The untokenized text of the second sequence.
    Only must be specified for sequence pair tasks.
    label: (Optional) string. The label of the example. This should be
    specified for train and dev examples, but not for test examples.
  """

  def __init__(self, guid, text_a, text_b=None, label=None, language=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.language = language

  def __repr__(self):
    return str(self.to_json_string())

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
  """
  A single set of features of data.
  Args:
    input_ids: Indices of input sequence tokens in the vocabulary.
    attention_mask: Mask to avoid performing attention on padding token indices.
      Mask values selected in ``[0, 1]``:
      Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
    token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    label: Label corresponding to the input
  """

  def __init__(self, input_ids, attention_mask=None, token_type_ids=None, langs=None, label=None):
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.label = label
    self.langs = langs

  def __repr__(self):
    return str(self.to_json_string())

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features_sc(
  examples,
  tokenizer,
  max_length=512,
  label_list=None,
  output_mode=None,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
  lang2id=None,
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    task: GLUE task
    label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
    output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  # is_tf_dataset = False
  # if is_tf_available() and isinstance(examples, tf.data.Dataset):
  #   is_tf_dataset = True

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))
    # if is_tf_dataset:
    #   example = processor.get_example_from_tensor_dict(example)
    #   example = processor.tfds_map(example)

    if isinstance(tokenizer, XLMTokenizer):
      inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, lang=example.language)
    else:
      inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length)
    
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
      input_ids = ([pad_token] * padding_length) + input_ids
      attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
      token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
      input_ids = input_ids + ([pad_token] * padding_length)
      attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
      token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    if lang2id is not None:
      lid = lang2id.get(example.language, lang2id["en"])
    else:
      lid = 0
    langs = [lid] * max_length

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
      len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
      len(token_type_ids), max_length
    )

    if output_mode == "classification":
      label = label_map[example.label]
    elif output_mode == "regression":
      label = float(example.label)
    else:
      raise KeyError(output_mode)

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      logger.info("label: %s (id = %d)" % (example.label, label))
      logger.info("language: %s, (lid = %d)" % (example.language, lid))

    features.append(
      InputFeatures(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, langs=langs, label=label
      )
    )
  return features





  