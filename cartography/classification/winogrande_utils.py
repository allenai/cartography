"""
Scripts to handle the Winogrande task.
"""
import json
import logging
import os
import tqdm

from transformers.data.processors.utils import DataProcessor

from cartography.classification.multiple_choice_utils import MCInputExample
from cartography.data_utils import read_data


class WinograndeProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
      return self._create_examples(
        read_data(os.path.join(data_dir, "train.tsv"), task_name="WINOGRANDE"))

    def get_dev_examples(self, data_dir):
      return self._create_examples(
        read_data(os.path.join(data_dir, "dev.tsv"), task_name="WINOGRANDE"))

    def get_test_examples(self, data_dir):
      return self._create_examples(
        read_data(os.path.join(data_dir, "test.tsv"), task_name="WINOGRANDE"))

    def get_examples(self, data_file, set_type):
      return self._create_examples(read_data(data_file, task_name="WINOGRANDE"))

    def get_labels(self):
      """See base class."""
      return ["1", "2"]

    def _build_example_from_named_fields(self, guid, sentence, name1, name2, label):
      conj = "_"
      idx = sentence.index(conj)
      context = sentence[:idx]
      option_str = "_ " + sentence[idx + len(conj):].strip()

      option1 = option_str.replace("_", name1)
      option2 = option_str.replace("_", name2)

      mc_example = MCInputExample(
          example_id=int(guid),
          contexts=[context, context],
          question=conj,
          endings = [option1, option2],
          label=label
      )
      return mc_example

    def _create_examples(self, records):
      tsv_dict, header = records
      examples = []
      for idx, line in tsv_dict.items():
        fields = line.strip().split("\t")
        assert idx == fields[0]
        sentence = fields[2]
        name1 = fields[3]
        name2 = fields[4]
        if len(fields) > 5:
          label = fields[-1]
        else:
          label = "1"  # Dummy label for test prediction.

        mc_example = self._build_example_from_named_fields(idx, sentence, name1, name2, label)
        examples.append(mc_example)

      return examples

    def _create_examples_jsonl(self, records):
      examples = []
      for (i, record) in enumerate(records):
        sentence = record['sentence']

        name1 = record['option1']
        name2 = record['option2']
        if not 'answer' in record:
          # This is a dummy label for test prediction.
          # test.jsonl doesn't include the `answer`.
          label = "1"
        else:
          label = record['answer']

        mc_example = self._build_example_from_named_fields(i, sentence, name1, name2, label)
        examples.append(mc_example)

      return examples
