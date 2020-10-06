"""
Randomly sample dataset examples for a data selection baseline.
"""
import argparse
import logging
import os
import pandas as pd

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

from cartography.data_utils import read_data, convert_tsv_entries_to_dataframe, copy_dev_test


if __name__ == "__main__":
  # Setup logging
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO,
  )
  parser = argparse.ArgumentParser()

  parser.add_argument("--input_dir",
                      "-i",
                      required=True,
                      type=os.path.abspath,
                      help="Path containing the TSV train file from which to subsample.",)
  parser.add_argument("--output_dir",
                      "-o",
                      required=True,
                      type=os.path.abspath,
                      help="Path where randomly subsampled data is to be written.",)
  parser.add_argument("--task_name",
                      "-t",
                      default="SNLI",
                      choices=("SNLI", "MNLI", "WINOGRANDE", "QNLI"),
                      help="Name of GLUE-style task.",)
  parser.add_argument("--seed",
                      type=int,
                      default=725862,
                      help="Random seed for sampling.")
  parser.add_argument("--fraction",
                      "-f",
                      type=float,
                      help="Number between 0 and 1, indicating fraction of random samples to select.")
  args = parser.parse_args()


  if args.fraction and 0 < args.fraction < 1:
    fractions = [args.fraction]
  else:
    fractions = [0.01, 0.05, 0.10, 0.1667, 0.25, 0.33, 0.50, 0.75]

  # Read the input train file.
  input_train_file = os.path.join(args.input_dir, "train.tsv")
  try:
    train = pd.read_csv(input_train_file, sep="\t")
  except pd.errors.ParserError:
    logger.info(f"Could not parse {input_train_file}. "
                 "Will read it as TSV and then convert into a Pandas dataframe.")
    train_dict, train_header = read_data(input_train_file, task_name=args.task_name)
    train = convert_tsv_entries_to_dataframe(train_dict, header=train_header)

  logger.info(f"Read {len(train)} examples from {input_train_file}. "
              f"Creating {fractions} subsamples...")
  outdir_base = f"{args.output_dir}_{args.seed}"
  for fraction in fractions:
    outdir = os.path.join(outdir_base, f"{args.task_name}_{fraction:.2f}/{args.task_name}")
    if not os.path.exists(outdir):
      os.makedirs(outdir)
    out_file_name = os.path.join(outdir, "train.tsv")

    # Dev and test need not be subsampled.
    copy_dev_test(args.task_name, from_dir=args.input_dir, to_dir=outdir)

    # Train set needs to be subsampled.
    train_sample = train.sample(n=int(fraction * len(train)),
                                random_state=args.seed)  # Set seed for replication.
    train_sample.to_csv(out_file_name, sep="\t", index=False)
    logger.info(f"Wrote {len(train_sample)} examples to {out_file_name}")
