from transformers.data.processors.glue import QnliProcessor

class AdaptedQnliProcessor(QnliProcessor):
  def get_examples(self, data_file, set_type):
      return self._create_examples(self._read_tsv(data_file), set_type=set_type)