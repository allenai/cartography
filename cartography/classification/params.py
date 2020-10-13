import json
import os


class Params:
  """
  All configuration required for running glue using transformers.
  """

  def __init__(self, MODEL_CLASSES, ALL_MODELS, processors, configs):

      ### Required parameters

      # Directory where task data resides.
      self.data_dir : str = configs["data_dir"]
      # Input data
      self.train : str = configs.get("train", None)
      self.dev : str = configs.get("dev", None)
      self.test: str = configs.get("test", None)

      # One of 'bert', 'roberta', etc.
      self.model_type : str = configs["model_type"]
      assert self.model_type in MODEL_CLASSES.keys()

      # Path to pre-trained model or shortcut name from `ALL_MODELS`.
      self.model_name_or_path : str  = configs["model_name_or_path"]
      assert self.model_name_or_path in ALL_MODELS

      # The name of the task to train.
      self.task_name : str = configs["task_name"]
      assert self.task_name.lower() in processors.keys()

      # Random seed for initialization.
      self.seed : int = configs["seed"]

      # The output directory where the model predictions and checkpoints will be written.
      self.output_dir : str = configs["output_dir"]

      # Whether to run training.
      self.do_train : bool = configs.get("do_train", False)

      # Whether to run eval on the dev set.
      self.do_eval : bool = configs.get("do_eval", False)

      # Whether to run eval on the dev set.
      self.do_test : bool = configs.get("do_test", False)

      ### Other parameters

      # Pretrained config name or path if not the same as `model_name`.
      self.config_name : str = configs.get("config_name", "")

      # Pretrained tokenizer name or path if not the same as `model_name`.
      self.tokenizer_name : str = configs.get("tokenizer_name", "")

      # Where to store the pre-trained models downloaded from s3:// location.
      self.cache_dir : str = configs.get("cache_dir", "")

      # Where to store the feature cache for the model.
      self.features_cache_dir : str = configs.get("features_cache_dir",
        os.path.join(self.data_dir, f"cache_{self.seed}"))

      # The maximum total input sequence length after tokenization.
      # Sequences longer than this will be truncated,
      # sequences shorter will be padded.
      self.max_seq_length : int = configs.get("max_seq_length", 128)

      # Run evaluation during training after each epoch.
      self.evaluate_during_training : bool = configs.get("evaluate_during_training", True)

      # Run evaluation during training at each logging step.
      self.evaluate_during_training_epoch : bool = configs.get("evaluate_during_training_epoch",
                                                               False)

      # Set this flag if you are using an uncased model.
      self.do_lower_case : bool =  configs.get("do_lower_case", True)

      # Batch size per GPU/CPU for training.
      self.per_gpu_train_batch_size : int = configs.get("per_gpu_train_batch_size", 96)

      # Batch size per GPU/CPU for evaluation.
      self.per_gpu_eval_batch_size : int = configs.get("per_gpu_eval_batch_size", 96)

      # Number of updates steps to accumulate before
      # performing a backward/update pass.
      self.gradient_accumulation_steps : int = configs.get("gradient_accumulation_steps", 1)

      # The initial learning rate for Adam.
      self.learning_rate : float = configs.get("learning_rate", 1e-5)

      # Weight decay if we apply some.
      self.weight_decay : float = configs.get("weight_decay", 0.0)

      # Epsilon for Adam optimizer.
      self.adam_epsilon : float = configs.get("adam_epsilon", 1e-8)

      # Max gradient norm.
      self.max_grad_norm : float = configs.get("max_grad_norm", 1.0)

      # Total number of training epochs to perform.
      self.num_train_epochs : float = configs.get("num_train_epochs", 3.0)

      # If > 0 : set total number of training steps to perform.
      # Override num_train_epochs.
      self.max_steps : int = configs.get("max_steps", -1)

      # Linear warmup over warmup_steps.
      self.warmup_steps : int = configs.get("warmup_steps", 0)

      # Log every X updates steps.
      self.logging_steps : int = configs.get("logging_steps", 1000)

      # If dev performance does not improve in X updates, end training.
      self.patience : int = configs.get("patience", 3)

      # Save checkpoint every X updates steps.
      self.save_steps:int = configs.get("save_steps", 0)

      # Evaluate all checkpoints starting with the same prefix as
      # model_name ending and ending with step number
      self.eval_all_checkpoints : bool = configs.get("eval_all_checkpoints", False)

      # Avoid using CUDA when available
      self.no_cuda : bool = configs.get("no_cuda", False)

      # Overwrite the content of the output directory
      self.overwrite_output_dir : bool = configs.get("overwrite_output_dir", False)

      # Overwrite the cached training and evaluation sets
      self.overwrite_cache : bool = configs.get("overwrite_cache", False)

      # Whether to use 16-bit (mixed) precision (through NVIDIA apex)
      # instead of 32-bit
      self.fp16 : bool = configs.get("fp16", False)

      # For fp16 : Apex AMP optimization level selected in
      # ['O0', 'O1', 'O2', and 'O3'].
      # See details at https://nvidia.github.io/apex/amp.html"
      self.fp16_opt_level : str = configs.get("fp16_opt_level", "01")

      # For distributed training.
      self.local_rank : int = configs.get("local_rank", -1)

      # For distant debugging.
      self.server_ip : str = configs.get("server_ip", "")
      self.server_port : str = configs.get("server_port", "")


def save_args_to_file(params: Params, mode: str = ""):
  """
  Saves the configs in `Params` to a json file, during train or eval mode.
  """
  with open(os.path.join(params.output_dir, f"cartography_config_{mode}.json"), "w") as outfile:
    writable_params = vars(params)

    # torch.device needs to be cast into a string to be json compatible.
    writable_params["device"] = str(params.device)

    outfile.write(json.dumps(writable_params, indent=4, sort_keys=True) + "\n")