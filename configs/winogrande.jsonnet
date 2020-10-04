local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_VISIBLE_DEVICES"));

local LEARNING_RATE = 1.1235456034244052e-05;
local BATCH_SIZE = 64;
local NUM_EPOCHS = 20;
local SEED = 71789;

local TASK = "WINOGRANDE";
local DATA_DIR = "/home/swabhas/data/glue/" + TASK + "/xl/" + TASK;
local FEATURES_CACHE_DIR = DATA_DIR + "/cache_" + SEED ;

local TEST = "/home/swabhas/data/glue/WINOGRANDE/wsc_superglue_trval_test.tsv";

{
   "data_dir": DATA_DIR,
   "model_type": "roberta_mc",
   "model_name_or_path": "roberta-large",
   "task_name": TASK,
   "seed": SEED,
   "num_train_epochs": NUM_EPOCHS,
   "learning_rate": LEARNING_RATE,
   "features_cache_dir": FEATURES_CACHE_DIR,
   "per_gpu_train_batch_size": BATCH_SIZE,
   "per_gpu_eval_batch_size": BATCH_SIZE,
   "do_train": true,
   "do_eval": true,
   "do_test": true,
   "test": TEST,
   "patience": 5,
}
