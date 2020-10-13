# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetuning the library models for sequence classification on GLUE-style tasks
(BERT, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa); modified for Dataset Cartography.
"""

import _jsonnet
import argparse
import glob
import json
import logging
import numpy as np
import os
import random
import shutil
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from cartography.classification.glue_utils import adapted_glue_compute_metrics as compute_metrics
from cartography.classification.glue_utils import adapted_glue_convert_examples_to_features as convert_examples_to_features
from cartography.classification.glue_utils import glue_output_modes as output_modes
from cartography.classification.glue_utils import glue_processors as processors
from cartography.classification.diagnostics_evaluation import evaluate_by_category
from cartography.classification.models import (
    AdaptedBertForMultipleChoice,
    AdaptedBertForSequenceClassification,
    AdaptedRobertaForMultipleChoice,
    AdaptedRobertaForSequenceClassification
)
from cartography.classification.multiple_choice_utils import convert_mc_examples_to_features
from cartography.classification.params import Params, save_args_to_file

from cartography.selection.selection_utils import log_training_dynamics


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            RobertaConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, AdaptedBertForSequenceClassification, BertTokenizer),
    "bert_mc": (BertConfig, AdaptedBertForMultipleChoice, BertTokenizer),
    "roberta": (RobertaConfig, AdaptedRobertaForSequenceClassification, RobertaTokenizer),
    "roberta_mc": (RobertaConfig, AdaptedRobertaForMultipleChoice, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_this_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_this_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info(f"  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {global_step}")
        logger.info(f"  Will skip the first {steps_trained_in_this_epoch} steps in the first epoch")

    tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained,
                            int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0],
                            mininterval=10,
                            ncols=100)
    set_seed(args)  # Added here for reproductibility
    best_dev_performance = 0
    best_epoch = epochs_trained

    train_acc = 0.0
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0],
                              mininterval=10,
                              ncols=100)

        train_iterator.set_description(f"train_epoch: {epoch} train_acc: {train_acc:.4f}")
        train_ids = None
        train_golds = None
        train_logits = None
        train_losses = None
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_this_epoch > 0:
                steps_trained_in_this_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if train_logits is None:  # Keep track of training dynamics.
                train_ids = batch[4].detach().cpu().numpy()
                train_logits = outputs[1].detach().cpu().numpy()
                train_golds = inputs["labels"].detach().cpu().numpy()
                train_losses = loss.detach().cpu().numpy()
            else:
                train_ids = np.append(train_ids, batch[4].detach().cpu().numpy())
                train_logits = np.append(train_logits, outputs[1].detach().cpu().numpy(), axis=0)
                train_golds = np.append(train_golds, inputs["labels"].detach().cpu().numpy())
                train_losses = np.append(train_losses, loss.detach().cpu().numpy())

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0] and
                    args.logging_steps > 0 and
                    global_step % args.logging_steps == 0
                ):
                    epoch_log = {}
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training_epoch:
                        logger.info(f"From within the epoch at step {step}")
                        results, _ = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            epoch_log[eval_key] = value

                    epoch_log["learning_rate"] = scheduler.get_lr()[0]
                    epoch_log["loss"] = (tr_loss - logging_loss) / args.logging_steps
                    logging_loss = tr_loss

                    for key, value in epoch_log.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**epoch_log, **{"step": global_step}}))

                if (
                    args.local_rank in [-1, 0] and
                    args.save_steps > 0 and
                    global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            epoch_iterator.set_description(f"lr = {scheduler.get_lr()[0]:.8f}, "
                                           f"loss = {(tr_loss-epoch_loss)/(step+1):.4f}")
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        #### Post epoch eval ####
        # Only evaluate when single GPU otherwise metrics may not average well
        if args.local_rank == -1 and args.evaluate_during_training:
            best_dev_performance, best_epoch = save_model(
                args, model, tokenizer, epoch, best_epoch, best_dev_performance)

        # Keep track of training dynamics.
        log_training_dynamics(output_dir=args.output_dir,
                              epoch=epoch,
                              train_ids=list(train_ids),
                              train_logits=list(train_logits),
                              train_golds=list(train_golds))
        train_result = compute_metrics(args.task_name, np.argmax(train_logits, axis=1), train_golds)
        train_acc = train_result["acc"]

        epoch_log = {"epoch": epoch,
                     "train_acc": train_acc,
                     "best_dev_performance": best_dev_performance,
                     "avg_batch_loss": (tr_loss - epoch_loss) / args.per_gpu_train_batch_size,
                     "learning_rate": scheduler.get_lr()[0],}
        epoch_loss = tr_loss

        logger.info(f"  End of epoch : {epoch}")
        with open(os.path.join(args.output_dir, f"eval_metrics_train.json"), "a") as toutfile:
            toutfile.write(json.dumps(epoch_log) + "\n")
        for key, value in epoch_log.items():
            tb_writer.add_scalar(key, value, global_step)
            logger.info(f"  {key}: {value:.6f}")

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        elif args.evaluate_during_training and epoch - best_epoch >= args.patience:
            logger.info(f"Ran out of patience. Best epoch was {best_epoch}. "
                f"Stopping training at epoch {epoch} out of {args.num_train_epochs} epochs.")
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def save_model(args, model, tokenizer, epoch, best_epoch,  best_dev_performance):
    results, _ = evaluate(args, model, tokenizer, prefix="in_training")
    # TODO(SS): change hard coding `acc` as the desired metric, might not work for all tasks.
    desired_metric = "acc"
    dev_performance = results.get(desired_metric)
    if dev_performance > best_dev_performance:
        best_epoch = epoch
        best_dev_performance = dev_performance

        # Save model checkpoint
        # Take care of distributed/parallel training
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        logger.info(f"*** Found BEST model, and saved checkpoint. "
            f"BEST dev performance : {dev_performance:.4f} ***")
    return best_dev_performance, best_epoch


def evaluate(args, model, tokenizer, prefix="", eval_split="dev"):
    # We do not really need a loop to handle MNLI double evaluation (matched, mis-matched).
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    all_predictions = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=True, data_split=f"{eval_split}_{prefix}")

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info(f"***** Running {eval_task} {prefix} evaluation on {eval_split} *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        example_ids = []
        gold_labels = []

        for batch in tqdm(eval_dataloader, desc="Evaluating", mininterval=10, ncols=100):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
                example_ids += batch[4].tolist()
                gold_labels += batch[3].tolist()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            probs = torch.nn.functional.softmax(torch.Tensor(preds), dim=-1)
            max_confidences = (torch.max(probs, dim=-1)[0]).tolist()
            preds = np.argmax(preds, axis=1)  # Max of logit is the same as max of probability.
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(
            eval_output_dir, f"eval_metrics_{eval_task}_{eval_split}_{prefix}.json")
        logger.info(f"***** {eval_task} {eval_split} results {prefix} *****")
        for key in sorted(result.keys()):
            logger.info(f"{eval_task} {eval_split} {prefix} {key} = {result[key]:.4f}")
        with open(output_eval_file, "a") as writer:
            writer.write(json.dumps(results) + "\n")

        # predictions
        all_predictions[eval_task] = []
        output_pred_file = os.path.join(
            eval_output_dir, f"predictions_{eval_task}_{eval_split}_{prefix}.lst")
        with open(output_pred_file, "w") as writer:
            logger.info(f"***** Write {eval_task} {eval_split} predictions {prefix} *****")
            for ex_id, pred, gold, max_conf, prob in zip(
                example_ids, preds, gold_labels, max_confidences, probs.tolist()):
                record = {"guid": ex_id,
                          "label": processors[args.task_name]().get_labels()[pred],
                          "gold": processors[args.task_name]().get_labels()[gold],
                          "confidence": max_conf,
                          "probabilities": prob}
                all_predictions[eval_task].append(record)
                writer.write(json.dumps(record) + "\n")

    return results, all_predictions


def load_dataset(args, task, eval_split="train"):
    processor = processors[task]()
    if eval_split == "train":
        if args.train is None:
            examples = processor.get_train_examples(args.data_dir)
        else:
            examples = processor.get_examples(args.train, "train")
    elif "dev" in eval_split:
        if args.dev is None:
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_examples(args.dev, "dev")
    elif "test" in eval_split:
        if args.test is None:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_examples(args.test, "test")
    else:
        raise ValueError(f"eval_split should be train / dev / test, but was given {eval_split}")

    return examples


def get_winogrande_tensors(features):
    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    # Convert to Tensors and build dataset
    input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)

    dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids, example_ids)
    return dataset


def load_and_cache_examples(args, task, tokenizer, evaluate=False, data_split="train"):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]

    if not os.path.exists(args.features_cache_dir):
        os.makedirs(args.features_cache_dir)
    cached_features_file = os.path.join(
        args.features_cache_dir,
        "cached_{}_{}_{}_{}".format(
            data_split,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    # Load data features from cache or dataset file
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = load_dataset(args, task, data_split)
        if task == "winogrande":
            features = convert_mc_examples_to_features(
                examples,
                label_list,
                args.max_seq_length,
                tokenizer,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,)
        else:
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                output_mode=output_mode,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training
        # process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if task == "winogrande":
        return get_winogrande_tensors(features)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_example_ids)
    return dataset


def run_transformer(args):
    if (os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty."
            f" Use --overwrite_output_dir to overcome.")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    args.learning_rate = float(args.learning_rate)
    if args.do_train:
        # If training for the first time, remove cache. If training from a checkpoint, keep cache.
        if os.path.exists(args.features_cache_dir) and not args.overwrite_output_dir:
            logger.info(f"Found existing cache for the same seed {args.seed}: "
                        f"{args.features_cache_dir}...Deleting!")
            shutil.rmtree(args.features_cache_dir)

        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
            save_args_to_file(args, mode="train")

        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss:.4f}")

    # Saving best-practices: if you use defaults names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        if not args.evaluate_during_training:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`

            # Take care of distributed/parallel training
            model_to_save = (model.module if hasattr(model, "module") else model)
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        logger.info(" **** Done with training ****")

    # Evaluation
    eval_splits = []
    if args.do_eval:
        eval_splits.append("dev")
    if args.do_test:
        eval_splits.append("test")

    if args.do_test or args.do_eval and args.local_rank in [-1, 0]:

        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        results = {}
        prefix = args.test.split("/")[-1].split(".tsv")[0] if args.test else ""
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix += checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            for eval_split in eval_splits:
                save_args_to_file(args, mode=eval_split)
                result, predictions = evaluate(args, model, tokenizer, prefix=prefix, eval_split=eval_split)
                result = dict((k + f"_{global_step}", v) for k, v in result.items())
                results.update(result)

            if args.test and "diagnostic" in args.test:
                # For running diagnostics with MNLI, run as SNLI and use hack.
                evaluate_by_category(predictions[args.task_name],
                                     mnli_hack=True if args.task_name in ["SNLI", "snli"] and "mnli" in args.output_dir else False,
                                     eval_filename=os.path.join(args.output_dir, f"eval_metrics_diagnostics.json"),
                                     diagnostics_file_carto=args.test)
    logger.info(" **** Done ****")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        "-c",
                        type=os.path.abspath,
                        required=True,
                        help="Main config file with basic arguments.")
    parser.add_argument("--output_dir",
                        "-o",
                        type=os.path.abspath,
                        required=True,
                        help="Output directory for model.")
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action="store_true",
                        help="Whether to run eval on the (OOD) test set.")
    parser.add_argument("--test",
                        type=os.path.abspath,
                        help="OOD test set.")

    # TODO(SS): Automatically map tasks to OOD test sets.

    args_from_cli = parser.parse_args()

    other_args = json.loads(_jsonnet.evaluate_file(args_from_cli.config))
    other_args.update(**vars(args_from_cli))
    args = Params(MODEL_CLASSES, ALL_MODELS, processors, other_args)
    run_transformer(args)


if __name__ == "__main__":
    main()
