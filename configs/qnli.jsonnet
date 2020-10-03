local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_VISIBLE_DEVICES"));

local LEARNING_RATE = 2.4721914839468448e-05;
local BATCH_SIZE = 96;
local NUM_EPOCHS = 20;
local SEED = 80965;

local TASK = "QNLI";
local DATA_DIR = "/home/swabhas/data/glue/" + TASK;
local FEATURES_CACHE_DIR = DATA_DIR + "/cache_" + SEED ;

{
   "data_dir": DATA_DIR,
   "model_type": "roberta",
   "model_name_or_path": "roberta-large",
   "task_name": TASK,
   "seed": SEED,
   "num_train_epochs": NUM_EPOCHS,
   "learning_rate": LEARNING_RATE,
   "features_cache_dir": FEATURES_CACHE_DIR,
   "per_gpu_train_batch_size": BATCH_SIZE
}
