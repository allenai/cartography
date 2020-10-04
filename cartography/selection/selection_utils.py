import json
import logging
import numpy as np
import os
import pandas as pd
import tqdm

from typing import List

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def log_training_dynamics(output_dir: os.path,
                          epoch: int,
                          train_ids: List[int],
                          train_logits: List[List[float]],
                          train_golds: List[int]):
  """
  Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
  """
  td_df = pd.DataFrame({"guid": train_ids,
                        f"logits_epoch_{epoch}": train_logits,
                        "gold": train_golds})

  logging_dir = os.path.join(output_dir, f"training_dynamics")
  # Create directory for logging training dynamics, if it doesn't already exist.
  if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
  epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
  td_df.to_json(epoch_file_name, lines=True, orient="records")
  logger.info(f"Training Dynamics logged to {epoch_file_name}")


def read_training_dynamics(model_dir: os.path,
                           strip_last: bool = False,
                           id_field: str = "guid"):
  """
  Given path to logged training dynamics, merge stats across epochs.
  Returns:
  - Dict between ID of a train instances and its gold label, and the list of logits across epochs.
  """
  train_dynamics = {}

  td_dir = os.path.join(model_dir, "training_dynamics/")
  num_epochs = len([f for f in os.listdir(td_dir) if os.path.isfile(os.path.join(td_dir, f))])

  logger.info(f"Reading {num_epochs} files from {td_dir} ...")
  for epoch_num in tqdm.tqdm(range(num_epochs)):
    epoch_file = os.path.join(td_dir, f"dynamics_epoch_{epoch_num}.json")
    assert os.path.exists(epoch_file)

    with open(epoch_file, "r") as infile:
      for line in infile:
        record = json.loads(line.strip())
        guid = record[id_field] if not strip_last else record[id_field][:-1]
        if guid not in train_dynamics:
          assert epoch_num == 0
          train_dynamics[guid] = {"gold": record["gold"], "logits": []}
        train_dynamics[guid]["logits"].append(record[f"logits_epoch_{epoch_num}"])

  logger.info(f"Found training dynamics for {len(train_dynamics)} train instances.")
  return train_dynamics


def save_forgetting_stats(forgetting_stats, data_dir, task_name):
  # Change data structure:
  logger.info(f"Size of forgetting stats: {len(forgetting_stats)}")
  stats = {}
  for epoch, pairs in forgetting_stats.items():
    if not len(pairs):
      break
    for pair in pairs:
      train_ids = pair[0].tolist()
      train_minibatch_results = pair[1]
      for tid, tresult in zip(train_ids, train_minibatch_results):
        if epoch == 0:
          assert tid not in stats
          stats[tid] = []
        stats[tid].append(bool(tresult))
  logger.info(f"Num examples: {len(stats)}")

  # Create directory for example forgetting.
  forgetting_dir = os.path.join(data_dir, f"forgetting_filtered/{task_name}")
  if not os.path.exists(forgetting_dir):
    os.makedirs(forgetting_dir)
  forgetting_stats_file = os.path.join(forgetting_dir, "forgetting_stats.json")

  # Dump forgetting trends into JSON file.
  with open(forgetting_stats_file, "w") as outfile:
    for key, trend in stats.items():
      outfile.write(json.dumps({"guid": key, "trend": trend}) + "\n")
  logger.info(f"Wrote forgetting stats to {forgetting_stats_file}")