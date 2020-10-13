import argparse
import logging
import numpy as np
import os

from collections import defaultdict
from sklearn.metrics import matthews_corrcoef

from cartography.data_utils_glue import read_glue_tsv, convert_string_to_unique_number

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Use the same fields as GLUE SNLI test + additional field for Diagnostic NLI.
FIELDS = ["index", "captionID", "pairID", "sentence1_binary_parse", "sentence2_binary_parse",
          "sentence1_parse", "sentence2_parse", "sentence1", "sentence2", "category", "gold_label",
          ]

LOGIC = ["Negation", "Double negation", "Intervals/Numbers", "Conjunction", "Disjunction",
          "Conditionals", "Universal", "Existential", "Temporal", "Upward monotone",
          "Downward monotone", "Non-monotone",
          ]
LEXSEM = ["Lexical entailment", "Morphological negation", "Factivity", "Symmetry/Collectivity",
            "Redundancy", "Named entities", "Quantifiers",
            ]
PAS = ["Core args", "Prepositional phrases", "Ellipsis/Implicits", "Anaphora/Coreference",
        "Active/Passive", "Nominalization", "Genitives/Partitives", "Datives", "Relative clauses",
        "Coordination scope", "Intersectivity", "Restrictivity",
        ]
KNOWLEDGE = ["Common sense", "World knowledge",
              ]


# Based on paper: https://openreview.net/pdf?id=rJ4km2R5t7
category_names = {"logic": 364,
                  "predicate_argument_structure": 424,
                  "lexical_semantics": 368,
                  "knowledge": 284}
coarse_to_fine = {"logic": LOGIC,
                  "predicate_argument_structure": PAS,
                  "lexical_semantics": LEXSEM,
                  "knowledge": KNOWLEDGE}

fine_to_coarse = {}
for coarse_cat, category in coarse_to_fine.items():
  for fine_cat in category:
    assert fine_cat not in fine_to_coarse
    fine_to_coarse[fine_cat] = coarse_cat


def label_balance(label_list):
  distribution = defaultdict(int)
  for label in label_list:
    distribution[label] += 1
  for label in distribution:
    distribution[label] /= len(label_list)
  return np.std(list(distribution.values()))


def determine_categories_by_fields(fields):
  example_categories = []
  for field in fields[:-4]:
    if field == '':
      continue
    elif ";" in field:
      example_categories.append(fine_to_coarse[field.split(";")[0]])  # Usually same coarse category.
    else:
      example_categories.append(fine_to_coarse[field])

  return example_categories


def diag_test_modifier(original_diag_tsv, output_tsv):
  """Modify the TSV file provided for Diagnostic NLI tests to follow the same
     format as the other test files for GLUE NLI."""
  diag_original, diag_headers = read_glue_tsv(original_diag_tsv, guid_index=None)
  coarse_category_counter = {name: 0 for name in category_names}

  with open(output_tsv, "w") as outfile:
    outfile.write("\t".join(FIELDS) + "\n")
    lines_with_missing_fields = 0
    multiple_categories = 0
    written = 0

    for i, (key, line) in enumerate(diag_original.items()):
      in_fields = line.strip().split("\t")

      if len(in_fields) < 8:
        # logger.info(f"Line with missing fields: {len(in_fields)} out of 8.\n  {in_fields}")
        lines_with_missing_fields += 1

      example_categories = determine_categories_by_fields(fields=in_fields)
      for ec in example_categories:
        coarse_category_counter[ec] += 1
      if len(example_categories) > 1:
        # logger.info(f"{len(category)} Categories : {category} \n {in_fields[:-4]}")
        multiple_categories += 1
      elif not len(example_categories):
        logger.info(f"No category found:\n {line}")
        # HACK: from my understanding, this is an example of factivity.
        example_categories = ["lexical_semantics"]

      guid = str(i)
      out_record = {"index": guid,
                    "captionID": guid,
                    "pairID": guid,
                    "sentence1_binary_parse": "",
                    "sentence2_binary_parse": "",
                    "sentence1_parse": "",
                    "sentence2_parse": "",
                    "gold_label": in_fields[-1],
                    "sentence2": in_fields[-2],
                    "sentence1": in_fields[-3],
                    "category": ";".join(example_categories)}
      out_fields = [out_record[field] if field in out_record else "" for field in FIELDS]
      outfile.write("\t".join(out_fields) + "\n")
      written += 1

  for c, count in coarse_category_counter.items():
    logger.info(f"Items in {c}: {count}")
    assert category_names[c] == count

  logger.info(f"Total records:               {len(diag_original)}")
  logger.info(f"Records with missing fields: {lines_with_missing_fields}.")
  logger.info(f"Records with 2+ categories:  {multiple_categories}.")
  logger.info(f"Total records written:       {written} to {output_tsv}")


def evaluate_by_category(predictions,
                         eval_filename,
                         mnli_hack = False,
                         diagnostics_file_carto = None,
                         diagnostics_file_original="/home/swabhas/diagnostics_nli/diagnostic-full.tsv"):
  if not diagnostics_file_carto and not os.path.exists(diagnostics_file_carto):
    diag_test_modifier(diagnostics_file_original, diagnostics_file_carto)

  diagnostics_orig, diag_headers = read_glue_tsv(diagnostics_file_carto)
  diagnostics = {convert_string_to_unique_number(key): val for key, val in diagnostics_orig.items()}

  # Category-wise counts.
  coarse_gold_labels = {key: [] for key in coarse_to_fine}
  coarse_predicted_labels = {key: [] for key in coarse_to_fine}

  # Some examples span multiple categories; maintain a global count to avoid overcounting.
  predicted_labels = []
  gold_labels = []

  if mnli_hack:  # Workaround for HuggingFace Transformers hack.
    logger.warning("WARNING: EMPLOYING HACK! "
                   "In HuggingFace Transformers, MNLI labels are swapped in the RoBERTa model."
                   "See: https://github.com/huggingface/transformers/blob/v2.8.0/examples/run_glue.py#L350"
                   "Hence, for evaluation, these need to be swapped back.")

  with open(eval_filename, "w") as outfile:
    for p in predictions:
      guid = p["guid"]
      if guid not in diagnostics:
          raise ValueError(f"Could not find predicted GUID: {p['guid']} in Diagnostic NLI test")

      gold_record = diagnostics[guid]
      gold_fields = gold_record.strip().split("\t")
      assert len(FIELDS) == len(gold_fields)

      gold_labels.append(gold_fields[-1])
      if mnli_hack:
        if p["label"] == "contradiction":
          p["label"] = "entailment"
        elif p["label"] == "entailment":
          p["label"] = "contradiction"
      predicted_labels.append(p["label"])

      diagnostic_categories = gold_fields[-2].split(";")
      for c in diagnostic_categories:
        coarse_gold_labels[c].append(gold_fields[-1])
        coarse_predicted_labels[c].append(p["label"])

    logged_results = []
    for cat, total in coarse_gold_labels.items():
      cat_r3 = matthews_corrcoef(y_true=coarse_gold_labels[cat], y_pred=coarse_predicted_labels[cat])
      cat_results = (np.array(coarse_gold_labels[cat]) == np.array(coarse_predicted_labels[cat]))

      cat_acc = cat_results.mean()
      logged_results.append(f"{cat:30}: {cat_acc:.4f} r3: {cat_r3:.4f}"
                            f" {cat_results.sum()}/{len(coarse_gold_labels[cat]):4}"
                            f" class-var: {label_balance(coarse_gold_labels[cat]):.3f}")

    overall_results = np.array(gold_labels) == np.array(predicted_labels)
    overall_accuracy = overall_results.mean()
    overall_r3 = matthews_corrcoef(y_true=gold_labels, y_pred=predicted_labels)

    logged_results.append(f"{'total acc':30}: {overall_accuracy:.4f} r3: {overall_r3:.4f}"
                          f" {overall_results.sum()}/{len(gold_labels):4}"
                          f" class-var: {label_balance(gold_labels):.3f}")

    for lr in logged_results:
      logger.info(lr)
      outfile.write(lr+"\n")

  logger.info(f"Results written to {eval_filename}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--diagnostics_input",
                      "-i",
                      type=os.path.abspath,
                      default="/home/swabhas/diagnostic_nli/diagnostic-full.tsv")
  parser.add_argument("--output",
                      "-o",
                      type=os.path.abspath,
                      default="/home/swabhas/data/glue/SNLI/diagnostics_test_bugfree.tsv")

  args = parser.parse_args()
  logger.info(args)

  diag_test_modifier(args.diagnostics_input, args.output)