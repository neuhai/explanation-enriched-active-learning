import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from tqdm.auto import tqdm
from transformers import (
    HfArgumentParser,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import (
    T2TDataCollator,
    batch_selection,
    prediction_model_preprocessing,
    rationale_model_preprocessing,
    single_model_preprocessing,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file_path: Optional[str] = field(
        default="train_data.pt",
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default="valid_data.pt",
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )


def predict(test_dataset, model_name, model_path, flag, iteration, config):
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(config["device"])
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    answers = []
    for batch in tqdm(dataloader):
        outs = model.generate(
            input_ids=batch["input_ids"].to(config["device"]),
            attention_mask=batch["attention_mask"].to(config["device"]),
            max_length=64,
            early_stopping=True,
        )
        outs = [
            tokenizer.decode(ids, skip_special_tokens=True)
            .encode("utf-8")
            .decode("utf-8")
            for ids in outs
        ]
        answers.extend(outs)

    predictions = []
    references = []
    for ref, pred in zip(test_dataset, answers):
        predictions.append(pred)
        # references.append(ref['answer'])

        references.append(tokenizer.decode(ref["target_ids"], skip_special_tokens=True))

    print("1st predicted:", predictions[0])
    print("1st groundtruth:", references[0])
    assert len(predictions) == len(references)
    # print(len(predictions), len(references))

    if model_name == "prediction":
        prediction_model_accuracy(predictions, references, flag, iteration, config)
    # elif model_name == 'rationale':
    #  rationale_model_accuracy(predictions, references)

    return predictions


def prediction_model_accuracy(predictions, references, flag, iteration, config):
    correct = total = 0

    for groundtruths, prediction in zip(references, predictions):
        total += 1

        if groundtruths == prediction:
            correct += 1
    accuracy = correct / total
    print("total test examples: ", total)
    print("prediction model accuracy: ", accuracy)
    if flag != "ignore":
        print(
            "RESULTS: [# current iteration %d], [# data per label %d], [# epoch RG %d], [# epoch P %d], [Learning Rate %f], [Batch size per device %d], [Select %s], [Accuracy %f]"
            % (
                iteration,
                config["num_data_per_batch"],
                config["num_epochs_rg"],
                config["num_epochs_p"],
                config["learning_rate"],
                config["per_device_batch_size"],
                flag,
                accuracy,
            )
        )


def finetune(config_json):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    # we will load the arguments from a json file,
    # make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_dict(config_json)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                training_args.output_dir
            )
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        # cache_dir=model_args.cache_dir,
    )
    tokenizer.add_special_tokens({"sep_token": "<sep>"})

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print("loading data")
    train_dataset = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    print("loading done")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results


def active_learning(
    criteria,
    train_dataset,
    sampled_test_dataset,
    rationale_dataset_test_dataset,
    config,
):
    batch_train_dataset = []
    remain_train_dataset = []
    previous_batch_train_dataset = []

    print("=== Active Learning with %s selector ===" % (criteria))

    for curr_iter in range(config["num_iter"]):
        print(
            "=== Start Batch Selection %d per label with %s ==="
            % (config["num_data_per_batch"], criteria)
        )
        # select a batch of data from train split
        if curr_iter == 0:
            if config["num_data_per_batch"] != -1:
                (
                    batch_train_dataset,
                    remain_train_dataset,
                    previous_batch_train_dataset,
                ) = batch_selection(
                    train_dataset,
                    config["num_data_per_batch"],
                    criteria,
                    0,
                    previous_batch_train_dataset,
                    config,
                )
            else:
                batch_train_dataset = train_dataset
        else:
            (
                batch_train_dataset,
                remain_train_dataset,
                previous_batch_train_dataset,
            ) = batch_selection(
                remain_train_dataset,
                config["num_data_per_batch"],
                criteria,
                curr_iter,
                previous_batch_train_dataset,
                config,
            )

        # previous_batch_train_dataset = copy.deepcopy(batch_train_dataset)

        print("=== Finish Batch Selection ===")
        print(
            "At Iteration %d, %d data selected, %d data remain "
            % (curr_iter, len(batch_train_dataset), len(remain_train_dataset))
        )

        # preprocess batch_train_dict
        rationale_model_batch_train_dataset = rationale_model_preprocessing(
            batch_train_dataset
        )
        torch.save(
            rationale_model_batch_train_dataset, "rationale_model_data/train_data.pt"
        )
        torch.save(
            rationale_model_batch_train_dataset, "rationale_model_data/valid_data.pt"
        )

        # fine-tune rationale model
        # set config and load model (1st time load pretrained model)
        rationale_model_config_json = {}
        if curr_iter == 0:
            rationale_model_config_json = {
                "model_name_or_path": "t5-base",
                "tokenizer_name": "t5-base",
                "max_len": 512,
                "target_max_len": 64,
                "train_file_path": "rationale_model_data/train_data.pt",
                "valid_file_path": "rationale_model_data/valid_data.pt",
                "output_dir": "rationale_model/",
                "overwrite_output_dir": True,
                "per_device_train_batch_size": config["per_device_batch_size"],
                "per_device_eval_batch_size": config["per_device_batch_size"],
                "gradient_accumulation_steps": 6,
                "learning_rate": config["learning_rate"],
                "num_train_epochs": config["num_epochs_rg"],
                "do_train": True,
                "do_eval": False,
                "prediction_loss_only": True,
                "remove_unused_columns": False,
                "save_strategy": "no",
                "evaluation_strategy": "no",
                "save_total_limit": 1,
                "load_best_model_at_end": True,
            }
        else:
            rationale_model_config_json = {
                "model_name_or_path": "rationale_model/",
                "tokenizer_name": "rationale_model/",
                "max_len": 512,
                "target_max_len": 64,
                "train_file_path": "rationale_model_data/train_data.pt",
                "valid_file_path": "rationale_model_data/valid_data.pt",
                "output_dir": "rationale_model/",
                "overwrite_output_dir": True,
                "per_device_train_batch_size": config["per_device_batch_size"],
                "per_device_eval_batch_size": config["per_device_batch_size"],
                "gradient_accumulation_steps": 6,
                "learning_rate": config["learning_rate"],
                "num_train_epochs": config["num_epochs_rg"],
                "do_train": True,
                "do_eval": False,
                "prediction_loss_only": True,
                "remove_unused_columns": False,
                "save_strategy": "no",
                "evaluation_strategy": "no",
                "save_total_limit": 1,
                "load_best_model_at_end": True,
            }

        # print("=== START FINETUNE MODEL RG ===")

        finetune(rationale_model_config_json)

        # print("=== START GENERATE RATIONALE FOR MODEL P ===")

        predicted_rationale = predict(
            rationale_model_batch_train_dataset,
            "rationale",
            "rationale_model/",
            "ignore",
            curr_iter,
            config,
        )

        # preprocess generated rationales
        prediction_model_batch_train_dataset = batch_train_dataset.add_column(
            "generated_rationale", predicted_rationale
        )
        prediction_model_batch_train_dataset = prediction_model_preprocessing(
            prediction_model_batch_train_dataset
        )
        torch.save(
            prediction_model_batch_train_dataset, "prediction_model_data/train_data.pt"
        )
        torch.save(
            prediction_model_batch_train_dataset, "prediction_model_data/valid_data.pt"
        )

        print("=== FINISH FINETUNE MODEL RG AT ITERATION %d ===" % (curr_iter))

        # fine-tune prediction model
        # set config and load model (1st time load pretrained model)
        prediction_model_config_json = {}
        if curr_iter == 0:
            prediction_model_config_json = {
                "model_name_or_path": "t5-base",
                "tokenizer_name": "t5-base",
                "max_len": 512,
                "target_max_len": 64,
                "train_file_path": "prediction_model_data/train_data.pt",
                "valid_file_path": "prediction_model_data/valid_data.pt",
                "output_dir": "prediction_model/",
                "overwrite_output_dir": True,
                "per_device_train_batch_size": config["per_device_batch_size"],
                "per_device_eval_batch_size": config["per_device_batch_size"],
                "gradient_accumulation_steps": 6,
                "learning_rate": 1e-4,
                "num_train_epochs": config["num_epochs_p"],
                "do_train": True,
                "do_eval": False,
                "prediction_loss_only": True,
                "remove_unused_columns": False,
                "save_strategy": "no",
                "evaluation_strategy": "no",
                "save_total_limit": 1,
                "load_best_model_at_end": True,
            }
        else:
            prediction_model_config_json = {
                "model_name_or_path": "prediction_model/",
                "tokenizer_name": "prediction_model/",
                "max_len": 512,
                "target_max_len": 64,
                "train_file_path": "prediction_model_data/train_data.pt",
                "valid_file_path": "prediction_model_data/valid_data.pt",
                "output_dir": "prediction_model/",
                "overwrite_output_dir": True,
                "per_device_train_batch_size": config["per_device_batch_size"],
                "per_device_eval_batch_size": config["per_device_batch_size"],
                "gradient_accumulation_steps": 6,
                "learning_rate": 1e-4,
                "num_train_epochs": config["num_epochs_p"],
                "do_train": True,
                "do_eval": False,
                "prediction_loss_only": True,
                "remove_unused_columns": False,
                "save_strategy": "no",
                "evaluation_strategy": "no",
                "save_total_limit": 1,
                "load_best_model_at_end": True,
            }

        # print("=== START FINETUNE MODEL P ===")

        finetune(prediction_model_config_json)

        print("=== FINISH FINETUNE MODEL P AT ITERATION %d ===" % (curr_iter))

        print("=== START EVALUATION AT ITERATION %d ===" % (curr_iter))

        predicted_rationale = predict(
            rationale_dataset_test_dataset,
            "rationale",
            "rationale_model/",
            "ignore",
            curr_iter,
            config,
        )
        prediction_model_test_dataset = sampled_test_dataset.add_column(
            "generated_rationale", predicted_rationale
        )
        prediction_model_test_dataset = prediction_model_preprocessing(
            prediction_model_test_dataset
        )
        predicted_model_prediction = predict(
            prediction_model_test_dataset,
            "prediction",
            "prediction_model/",
            criteria,
            curr_iter,
            config,
        )

        # GARBAGE COLLECTION
        del prediction_model_test_dataset
        del predicted_rationale
        del rationale_model_batch_train_dataset
        del prediction_model_batch_train_dataset


def active_learning_uncertainty(
    criteria,
    train_dataset,
    sampled_test_dataset,
    rationale_dataset_test_dataset,
    config,
):
    batch_train_dataset = []
    remain_train_dataset = []
    previous_batch_train_dataset = []

    print("=== Active Learning with %s selector ===" % (criteria))

    for curr_iter in range(config["num_iter"]):
        print(
            "=== Start Batch Selection %d per label with %s ==="
            % (config["num_data_per_batch"], criteria)
        )
        # select a batch of data from train split
        if curr_iter == 0:
            if config["num_data_per_batch"] != -1:
                (
                    batch_train_dataset,
                    remain_train_dataset,
                    previous_batch_train_dataset,
                ) = batch_selection(
                    train_dataset,
                    config["num_data_per_batch"],
                    criteria,
                    0,
                    previous_batch_train_dataset,
                    config,
                )
            else:
                batch_train_dataset = train_dataset
        else:
            (
                batch_train_dataset,
                remain_train_dataset,
                previous_batch_train_dataset,
            ) = batch_selection(
                remain_train_dataset,
                config["num_data_per_batch"],
                criteria,
                curr_iter,
                previous_batch_train_dataset,
                config,
            )

        # previous_batch_train_dataset = copy.deepcopy(batch_train_dataset)

        print("=== Finish Batch Selection ===")
        print(
            "At Iteration %d, %d data selected, %d data remain "
            % (curr_iter, len(batch_train_dataset), len(remain_train_dataset))
        )
        if "rationale" in criteria:
            # preprocess batch_train_dict
            rationale_model_batch_train_dataset = rationale_model_preprocessing(
                batch_train_dataset
            )
            torch.save(
                rationale_model_batch_train_dataset,
                "rationale_model_data/train_data.pt",
            )
            torch.save(
                rationale_model_batch_train_dataset,
                "rationale_model_data/valid_data.pt",
            )

            # fine-tune rationale model
            # set config and load model (1st time load pretrained model)
            rationale_model_config_json = {}
            if curr_iter == 0:
                rationale_model_config_json = {
                    "model_name_or_path": "t5-base",
                    "tokenizer_name": "t5-base",
                    "max_len": 512,
                    "target_max_len": 64,
                    "train_file_path": "rationale_model_data/train_data.pt",
                    "valid_file_path": "rationale_model_data/valid_data.pt",
                    "output_dir": "rationale_model/",
                    "overwrite_output_dir": True,
                    "per_device_train_batch_size": config["per_device_batch_size"],
                    "per_device_eval_batch_size": config["per_device_batch_size"],
                    "gradient_accumulation_steps": 6,
                    "learning_rate": config["learning_rate"],
                    "num_train_epochs": config["num_epochs_rg"],
                    "do_train": True,
                    "do_eval": False,
                    "prediction_loss_only": True,
                    "remove_unused_columns": False,
                    "save_strategy": "no",
                    "evaluation_strategy": "no",
                    "save_total_limit": 1,
                    "load_best_model_at_end": True,
                }
            else:
                rationale_model_config_json = {
                    "model_name_or_path": "rationale_model/",
                    "tokenizer_name": "rationale_model/",
                    "max_len": 512,
                    "target_max_len": 64,
                    "train_file_path": "rationale_model_data/train_data.pt",
                    "valid_file_path": "rationale_model_data/valid_data.pt",
                    "output_dir": "rationale_model/",
                    "overwrite_output_dir": True,
                    "per_device_train_batch_size": config["per_device_batch_size"],
                    "per_device_eval_batch_size": config["per_device_batch_size"],
                    "gradient_accumulation_steps": 6,
                    "learning_rate": config["learning_rate"],
                    "num_train_epochs": config["num_epochs_rg"],
                    "do_train": True,
                    "do_eval": False,
                    "prediction_loss_only": True,
                    "remove_unused_columns": False,
                    "save_strategy": "no",
                    "evaluation_strategy": "no",
                    "save_total_limit": 1,
                    "load_best_model_at_end": True,
                }

            # print("=== START FINETUNE MODEL RG ===")

            finetune(rationale_model_config_json)

            # print("=== START GENERATE RATIONALE FOR MODEL P ===")

            predicted_rationale = predict(
                rationale_model_batch_train_dataset,
                "rationale",
                "rationale_model/",
                "ignore",
                curr_iter,
                config,
            )

            # preprocess generated rationales
            prediction_model_batch_train_dataset = batch_train_dataset.add_column(
                "generated_rationale", predicted_rationale
            )
            prediction_model_batch_train_dataset = prediction_model_preprocessing(
                prediction_model_batch_train_dataset
            )
            torch.save(
                prediction_model_batch_train_dataset,
                "prediction_model_data/train_data.pt",
            )
            torch.save(
                prediction_model_batch_train_dataset,
                "prediction_model_data/valid_data.pt",
            )
        else:
            prediction_model_batch_train_dataset = single_model_preprocessing(
                batch_train_dataset
            )
            torch.save(
                prediction_model_batch_train_dataset,
                "prediction_model_data/train_data.pt",
            )
            torch.save(
                prediction_model_batch_train_dataset,
                "prediction_model_data/valid_data.pt",
            )

            print("=== FINISH FINETUNE MODEL RG AT ITERATION %d ===" % (curr_iter))

        # fine-tune prediction model
        # set config and load model (1st time load pretrained model)
        prediction_model_config_json = {}
        if curr_iter == 0:
            prediction_model_config_json = {
                "model_name_or_path": "t5-base",
                "tokenizer_name": "t5-base",
                "max_len": 512,
                "target_max_len": 64,
                "train_file_path": "prediction_model_data/train_data.pt",
                "valid_file_path": "prediction_model_data/valid_data.pt",
                "output_dir": "prediction_model/",
                "overwrite_output_dir": True,
                "per_device_train_batch_size": config["per_device_batch_size"],
                "per_device_eval_batch_size": config["per_device_batch_size"],
                "gradient_accumulation_steps": 6,
                "learning_rate": 1e-4,
                "num_train_epochs": config["num_epochs_p"],
                "do_train": True,
                "do_eval": False,
                "prediction_loss_only": True,
                "remove_unused_columns": False,
                "save_strategy": "no",
                "evaluation_strategy": "no",
                "save_total_limit": 1,
                "load_best_model_at_end": True,
            }
        else:
            prediction_model_config_json = {
                "model_name_or_path": "prediction_model/",
                "tokenizer_name": "prediction_model/",
                "max_len": 512,
                "target_max_len": 64,
                "train_file_path": "prediction_model_data/train_data.pt",
                "valid_file_path": "prediction_model_data/valid_data.pt",
                "output_dir": "prediction_model/",
                "overwrite_output_dir": True,
                "per_device_train_batch_size": config["per_device_batch_size"],
                "per_device_eval_batch_size": config["per_device_batch_size"],
                "gradient_accumulation_steps": 6,
                "learning_rate": 1e-4,
                "num_train_epochs": config["num_epochs_p"],
                "do_train": True,
                "do_eval": False,
                "prediction_loss_only": True,
                "remove_unused_columns": False,
                "save_strategy": "no",
                "evaluation_strategy": "no",
                "save_total_limit": 1,
                "load_best_model_at_end": True,
            }

        # print("=== START FINETUNE MODEL P ===")

        finetune(prediction_model_config_json)

        print("=== FINISH FINETUNE MODEL P AT ITERATION %d ===" % (curr_iter))

        print("=== START EVALUATION AT ITERATION %d ===" % (curr_iter))

        if "rationale" in criteria:
            predicted_rationale = predict(
                rationale_dataset_test_dataset,
                "rationale",
                "rationale_model/",
                "ignore",
                curr_iter,
                config,
            )
            prediction_model_test_dataset = sampled_test_dataset.add_column(
                "generated_rationale", predicted_rationale
            )
            prediction_model_test_dataset = prediction_model_preprocessing(
                prediction_model_test_dataset
            )
        else:
            prediction_model_test_dataset = single_model_preprocessing(
                rationale_dataset_test_dataset
            )

        predicted_model_prediction = predict(
            prediction_model_test_dataset,
            "prediction",
            "prediction_model/",
            criteria,
            curr_iter,
            config,
        )

        # GARBAGE COLLECTION
        del prediction_model_test_dataset
        if "rationale" in criteria:
            del predicted_rationale
            del rationale_model_batch_train_dataset
        del prediction_model_batch_train_dataset
