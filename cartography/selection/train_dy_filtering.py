"""
Filtering methods based on training dynamics.
"""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import tqdm

from collections import defaultdict
from typing import List

from cartography.data_utils import read_data, read_jsonl, copy_dev_test
from cartography.selection.selection_utils import read_training_dynamics

# TODO(SS): Named tuple for tasks and filtering methods.

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def compute_forgetfulness(correctness_trend: List[float]) -> int:
  """
  Given a epoch-wise trend of train predictions, compute frequency with which
  an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
  Based on: https://arxiv.org/abs/1812.05159
  """
  if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
      return 1000
  learnt = False  # Predicted correctly in the current epoch.
  times_forgotten = 0
  for is_correct in correctness_trend:
    if (not learnt and not is_correct) or (learnt and is_correct):
      # nothing changed.
      continue
    elif learnt and not is_correct:
      # Forgot after learning at some point!
      learnt = False
      times_forgotten += 1
    elif not learnt and is_correct:
      # Learnt!
      learnt = True
  return times_forgotten


def compute_correctness(trend: List[float]) -> float:
  """
  Aggregate #times an example is predicted correctly during all training epochs.
  """
  return sum(trend)


def compute_train_dy_metrics(training_dynamics, args):
  """
  Given the training dynamics (logits for each training instance across epochs), compute metrics
  based on it, for data map coorodinates.
  Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
  the last two being baselines from prior work
  (Example Forgetting: https://arxiv.org/abs/1812.05159 and
   Active Bias: https://arxiv.org/abs/1704.07433 respectively).
  Returns:
  - DataFrame with these metrics.
  - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
  """
  confidence_ = {}
  variability_ = {}
  threshold_closeness_ = {}
  correctness_ = {}
  forgetfulness_ = {}

  # Functions to be applied to the data.
  variability_func = lambda conf: np.std(conf)
  if args.include_ci:  # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
    variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
  threshold_closeness_func = lambda conf: conf * (1 - conf)

  loss = torch.nn.CrossEntropyLoss()

  num_tot_epochs = len(list(training_dynamics.values())[0]["logits"])
  if args.burn_out < num_tot_epochs:
    logger.info(f"Computing training dynamics. Burning out at {args.burn_out} of {num_tot_epochs}. ")
  else:
    logger.info(f"Computing training dynamics across {num_tot_epochs}")
  logger.info("Metrics computed: confidence, variability, correctness, forgetfulness, threshold_closeness")

  logits = {i: [] for i in range(num_tot_epochs)}
  targets = {i: [] for i in range(num_tot_epochs)}
  training_accuracy = defaultdict(float)

  for guid in tqdm.tqdm(training_dynamics):
    correctness_trend = []
    true_probs_trend = []

    record = training_dynamics[guid]
    for i, epoch_logits in enumerate(record["logits"]):
      probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
      true_class_prob = float(probs[record["gold"]])
      true_probs_trend.append(true_class_prob)

      prediction = np.argmax(epoch_logits)
      is_correct = (prediction == record["gold"]).item()
      correctness_trend.append(is_correct)

      training_accuracy[i] += is_correct
      logits[i].append(epoch_logits)
      targets[i].append(record["gold"])

    if args.burn_out < num_tot_epochs:
      correctness_trend = correctness_trend[:args.burn_out]
      true_probs_trend = true_probs_trend[:args.burn_out]

    correctness_[guid] = compute_correctness(correctness_trend)
    confidence_[guid] = np.mean(true_probs_trend)
    variability_[guid] = variability_func(true_probs_trend)

    forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
    threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])

  # Should not affect ranking, so ignoring.
  epsilon_var = np.mean(list(variability_.values()))

  column_names = ['guid',
                  'index',
                  'threshold_closeness',
                  'confidence',
                  'variability',
                  'correctness',
                  'forgotfulness',]
  df = pd.DataFrame([[guid,
                      i,
                      threshold_closeness_[guid],
                      confidence_[guid],
                      variability_[guid],
                      correctness_[guid],
                      forgetfulness_[guid],
                      ] for i, guid in enumerate(correctness_)], columns=column_names)

  df_train = pd.DataFrame([[i,
                            loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(training_dynamics),
                            training_accuracy[i] / len(training_dynamics)
                            ] for i in range(num_tot_epochs)],
                          columns=['epoch', 'loss', 'train_acc'])
  return df, df_train


def consider_ascending_order(filtering_metric: str) -> bool:
  """
  Determine if the metric values' sorting order to get the most `valuable` examples for training.
  """
  if filtering_metric == "variability":
    return False
  elif filtering_metric == "confidence":
    return True
  elif filtering_metric == "threshold_closeness":
    return False
  elif filtering_metric == "forgetfulness":
    return False
  elif filtering_metric == "correctness":
    return True
  else:
    raise NotImplementedError(f"Filtering based on {filtering_metric} not implemented!")


def write_filtered_data(args, train_dy_metrics):
  """
  Filter data based on the given metric, and write it in TSV format to train GLUE-style classifier.
  """
  # Writing the data --- CHANGE all 3 below: name, index, and sorting order (reverse or not.)
  is_ascending = consider_ascending_order(args.metric)

  # Apply selections.
  if args.worst:
    is_ascending = not is_ascending
  sorted_scores = train_dy_metrics.sort_values(by=[args.metric],
                                               ascending=is_ascending)

  train_file_name = os.path.join(os.path.join(args.data_dir, args.task_name), f"train.tsv")
  train_numeric, header = read_data(train_file_name, task_name=args.task_name, guid_as_int=True)

  for fraction in [0.01, 0.05, 0.10, 0.1667, 0.25, 0.3319, 0.50, 0.75]:
    outdir = os.path.join(args.filtering_output_dir,
                          f"cartography_{args.metric}_{fraction:.2f}/{args.task_name}")
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    # Dev and test need not be subsampled.
    copy_dev_test(args.task_name,
                  from_dir=os.path.join(args.data_dir, args.task_name),
                  to_dir=outdir)

    num_samples = int(fraction * len(train_numeric))
    with open(os.path.join(outdir, f"train.tsv"), "w") as outfile:
      outfile.write(header + "\n")
      selected = sorted_scores.head(n=num_samples+1)
      if args.both_ends:
        hardest = sorted_scores.head(n=int(num_samples * 0.7))
        easiest = sorted_scores.tail(n=num_samples - hardest.shape[0])
        selected = pd.concat([hardest, easiest])
        fm = args.metric
        logger.info(f"Selecting both ends: {fm} = "
                    f"({hardest.head(1)[fm].values[0]:3f}: {hardest.tail(1)[fm].values[0]:3f}) "
                    f"& ({easiest.head(1)[fm].values[0]:3f}: {easiest.tail(1)[fm].values[0]:3f})")

      selection_iterator = tqdm.tqdm(range(len(selected)))
      for idx in selection_iterator:
        selection_iterator.set_description(
          f"{args.metric} = {selected.iloc[idx][args.metric]:.4f}")

        selected_id = selected.iloc[idx]["guid"]
        if args.task_name in ["SNLI", "MNLI"]:
          selected_id = int(selected_id)
        record = train_numeric[selected_id]
        outfile.write(record + "\n")

    logger.info(f"Wrote {num_samples} samples to {outdir}.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode',
                      choices=('display', 'filter', 'both', 'neither'),
                      required=True,
                      help="Whether to display scatter plots, filter subsets or both.")
  parser.add_argument("--model_dir",
                      "-o",
                      required=True,
                      type=os.path.abspath,
                      help="Directory where model training dynamics stats reside.")
  parser.add_argument("--data_dir",
                      "-d",
                      default="/Users/swabhas/data/glue/WINOGRANDE/xl/",
                      type=os.path.abspath,
                      help="Directory where data for task resides.")
  parser.add_argument("--plots_dir",
                      default="/Users/swabhas/Documents/exemplars_plots/",
                      type=os.path.abspath,
                      help="Directory where plots are to be saved.")
  parser.add_argument("--task_name",
                      "-t",
                      default="WINOGRANDE",
                      choices=("SNLI", "MNLI", "QNLI", "WINOGRANDE"),
                      help="Which task are we plotting or filtering for.")
  parser.add_argument('--metric',
                      choices=('threshold_closeness',
                               'confidence',
                               'variability',
                               'correctness',
                               'forgetfulness'),
                      default='variability',
                      help="Metric to filter data by.",)
  parser.add_argument("--include_ci",
                      action="store_true",
                      help="Compute the confidence interval for variability.")
  parser.add_argument("--filtering_output_dir",
                      "-f",
                      default="/Users/swabhas/data/exemplars/variability_filtered/",
                      type=os.path.abspath,
                      help="Output directory where filtered datasets are to be written.")
  parser.add_argument("--worst",
                      action="store_true",
                      help="Select from the opposite end of the spectrum acc. to metric,"
                           "for baselines")
  parser.add_argument("--both_ends",
                      action="store_true",
                      help="Select from both ends of the spectrum acc. to metric,")
  parser.add_argument("--burn_out",
                      type=int,
                      default=100,
                      help="# Epochs for which to compute train dynamics.")

  args = parser.parse_args()

  training_dynamics = read_training_dynamics(args.model_dir,
                                             strip_last=True if args.task_name in ["QNLI"] else False)
  total_epochs = len(list(training_dynamics.values())[0]["logits"])
  if args.burn_out > total_epochs:
    args.burn_out = total_epochs
    logger.info(f"Total epochs found: {args.burn_out}")
  train_dy_metrics, _ = compute_train_dy_metrics(training_dynamics, args)

  burn_out_str = f"_{args.burn_out}" if args.burn_out > total_epochs else ""
  train_dy_filename = os.path.join(args.model_dir, f"td_metrics{burn_out_str}.jsonl")
  train_dy_metrics.to_json(train_dy_filename,
                           orient='records',
                           lines=True)
  logger.info(f"Metrics based on Training Dynamics written to {train_dy_filename}")

  if args.mode == 'display' or args.mode == 'both':
    model_name = args.model_dir.split("log_")[-1]
    outdir = os.path.join(args.plots_dir, f"{args.task_name.lower()}_{model_name}")
    if not os.path.exists(outdir):
      os.makedirs(outdir)

  if args.mode == 'filter' or args.mode == 'both':
    write_filtered_data(args, train_dy_metrics)
