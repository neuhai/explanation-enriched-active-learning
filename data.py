import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)

tokenizer = T5Tokenizer.from_pretrained("t5-base")
tokenizer.add_special_tokens({"sep_token": "<sep>"})

model_SentenceTransformer = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


label = ["entailment", "neutral", "contradiction"]


# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T2TDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """

        input_ids = torch.stack([example["input_ids"] for example in batch])
        labels = torch.stack([example["target_ids"] for example in batch])
        labels[labels[:, :] == 0] = -100
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["target_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask,
        }


def get_scores(test_dataset, model_path, config):
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(config["device"])
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    scores = []
    for batch in tqdm(dataloader):
        outs = model.generate(
            input_ids=batch["input_ids"].to(config["device"]),
            attention_mask=batch["attention_mask"].to(config["device"]),
            max_length=64,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
        )
        scores.append(
            model.compute_transition_scores(
                outs.sequences, outs.scores, normalize_logits=True
            )[:, 0]
        )
    scores = torch.cat(scores, dim=0)
    return scores


def predict_rationale(test_dataset, model_path, config):
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
    return answers


def rationale_model_define_input(example):
    label_content = ""
    for i in range(len(label)):
        label_content += "choice" + str(i + 1) + ": " + str(label[i]) + " "

    example["input_text"] = (
        "explain: what is the relationship between %s and %s "
        % (example["hypothesis"], example["premise"])
    ) + label_content
    example["target_text"] = "%s" % example["explanation_1"]

    return example


def prediction_model_define_input(example):
    label_content = ""
    for i in range(len(label)):
        label_content += "choice" + str(i + 1) + ": " + str(label[i]) + " "

    example["input_text"] = (
        (
            "question: what is the relationship between %s and %s "
            % (example["hypothesis"], example["premise"])
        )
        + label_content
        + (" <sep> because %s" % (example["generated_rationale"]))
    )
    example["target_text"] = "%s" % label[int(example["label"])]

    return example


def single_model_define_input(example):
    label_content = ""
    for i in range(len(label)):
        label_content += "choice" + str(i + 1) + ": " + str(label[i]) + " "

    example["input_text"] = (
        (
            "question: what is the relationship between %s and %s "
            % (example["hypothesis"], example["premise"])
        )
        + label_content
        + (" <sep>")
    )
    example["target_text"] = "%s" % label[int(example["label"])]

    return example


def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(
        example_batch["input_text"],
        add_special_tokens=True,
        truncation=True,
        pad_to_max_length=True,
        max_length=512,
    )
    target_encodings = tokenizer.batch_encode_plus(
        example_batch["target_text"],
        add_special_tokens=True,
        truncation=True,
        pad_to_max_length=True,
        max_length=64,
    )

    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "target_ids": target_encodings["input_ids"],
        "target_attention_mask": target_encodings["attention_mask"],
    }

    return encodings


def rationale_model_preprocessing(input_dataset):
    input_dataset = input_dataset.map(
        rationale_model_define_input, load_from_cache_file=False
    )

    # print(input_dataset[0])

    input_dataset = input_dataset.map(
        convert_to_features, batched=True, load_from_cache_file=False
    )

    columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
    input_dataset.set_format(type="torch", columns=columns)
    return input_dataset


def prediction_model_preprocessing(input_dataset):
    input_dataset = input_dataset.map(
        prediction_model_define_input, load_from_cache_file=False
    )
    input_dataset = input_dataset.map(
        convert_to_features, batched=True, load_from_cache_file=False
    )

    columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
    input_dataset.set_format(type="torch", columns=columns)
    return input_dataset


def single_model_preprocessing(input_dataset):
    input_dataset = input_dataset.map(
        single_model_define_input, load_from_cache_file=False
    )
    input_dataset = input_dataset.map(
        convert_to_features, batched=True, load_from_cache_file=False
    )

    columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
    input_dataset.set_format(type="torch", columns=columns)
    return input_dataset


def custom_concatenate(dataset1, dataset2, dataset3):
    new_dataset = {}

    for key in dataset1.features.keys():
        if key != "Unnamed: 0":
            new_dataset[key] = []

    for example in dataset1:
        for key in example:
            if key != "Unnamed: 0":
                new_dataset[key].append(example[key])

    for example in dataset2:
        for key in example:
            if key != "Unnamed: 0":
                new_dataset[key].append(example[key])

    for example in dataset3:
        for key in example:
            if key != "Unnamed: 0":
                new_dataset[key].append(example[key])

    new_dataset = Dataset.from_dict(new_dataset)

    return new_dataset


def custom_concatenate_2(dataset1, dataset2):
    new_dataset = {}

    for key in dataset1.features.keys():
        if key != "Unnamed: 0":
            new_dataset[key] = []

    for example in dataset1:
        for key in example:
            if key != "Unnamed: 0":
                new_dataset[key].append(example[key])

    for example in dataset2:
        for key in example:
            if key != "Unnamed: 0":
                new_dataset[key].append(example[key])

    new_dataset = Dataset.from_dict(new_dataset)

    return new_dataset


def random_batch_selection(train_dataset, num_batch):
    # sampled_idx = random.sample( list( range(len(train_dataset)) ), num_batch )
    sampled_idx = torch.randint(0, len(train_dataset), (num_batch,))

    # remaining_idx = list( range(len(train_dataset)) )
    # for i in sampled_idx:
    #   remaining_idx.remove(i)

    # improve speed
    remaining_idx = torch.arange(len(train_dataset))
    ones = torch.ones_like(remaining_idx, dtype=torch.bool)
    ones[sampled_idx] = False
    remaining_idx = remaining_idx[ones]
    # batch_dataset = train_dataset.select(sampled_idx)
    # remain_dataset = train_dataset.select(remaining_idx)
    # return batch_dataset, remain_dataset
    return train_dataset.select(sampled_idx), train_dataset.select(remaining_idx)


def uncertainty_batch_selection(
    train_dataset, num_batch, criteria, current_iter, config
):
    # THIS METHOD ONLY COMPARE THE TASK CONTENT
    if criteria == "uncertainty":
        # During the first iteration, there's no model prediction
        if current_iter == 0:
            return random_batch_selection(train_dataset, num_batch)
        dataset = single_model_preprocessing(train_dataset)
        scores = get_scores(dataset, "prediction_model/", config)
        # score higher is better
        # select lower score
        sampled_idx = scores.argsort()[:num_batch]
        remaining_idx = scores.argsort()[num_batch:]

    # TTHIS METHOD COMPARE THE RATIONALES AND TASK CONTENT
    elif criteria == "uncertainty_rationale":
        # During the first iteration, there's no model prediction
        if current_iter == 0:
            return random_batch_selection(train_dataset, num_batch)
        predicted_rationale = predict_rationale(
            rationale_model_preprocessing(train_dataset), "rationale_model/", config
        )
        prediction_model_test_dataset = train_dataset.add_column(
            "generated_rationale", predicted_rationale
        )
        prediction_model_test_dataset = prediction_model_preprocessing(
            prediction_model_test_dataset
        )
        scores = get_scores(prediction_model_test_dataset, "prediction_model/", config)
        # score higher is better
        # select lower score
        sampled_idx = scores.argsort()[:num_batch]
        remaining_idx = scores.argsort()[num_batch:]

    remaining_idx = torch.arange(len(train_dataset))
    ones = torch.ones_like(remaining_idx, dtype=torch.bool)
    ones[sampled_idx] = False
    remaining_idx = remaining_idx[ones]
    return train_dataset.select(sampled_idx), train_dataset.select(remaining_idx)


def similarity_batch_selection(
    train_dataset,
    num_batch,
    criteria,
    current_iter,
    previous_batch_train_dataset,
    config,
):
    # THESE TWO METHODS ONLY COMPARE THE TASK CONTENT
    if criteria == "combined" or criteria == "even":
        # During the first iteration, there's no human annotattion
        # So we compare the similarity between original content
        if current_iter == 0:
            sentence_list = []
            rationale_list = []
            for i in train_dataset:
                sentence_list.append(i["hypothesis"] + " " + i["premise"])
                rationale_list.append(i["hypothesis"] + " " + i["premise"])
        else:
            sentence_list = []
            rationale_list = []
            for i in train_dataset:
                sentence_list.append(i["hypothesis"] + " " + i["premise"])
            for i in previous_batch_train_dataset:
                rationale_list.append(i["hypothesis"] + " " + i["premise"])

        sentence_list_embeddings = model_SentenceTransformer.encode(
            sentence_list, convert_to_tensor=True
        ).to(config["device"])
        rationale_list_embeddings = model_SentenceTransformer.encode(
            rationale_list, convert_to_tensor=True
        ).to(config["device"])

        cosine_scores = util.cos_sim(
            sentence_list_embeddings, rationale_list_embeddings
        )

        mean_cosine_scores = torch.mean(cosine_scores, 1, True)
        # mean_cosine_scores_list = [i.item() for i in mean_cosine_scores]
        # mean_cosine_scores_list_sorted_with_idx = sorted(range(len(mean_cosine_scores_list)), key=lambda i: mean_cosine_scores_list[i])
        # improve speed
        mean_cosine_scores_list_sorted_with_idx = mean_cosine_scores.squeeze(
            1
        ).argsort()

        # combine half top ranked ones and half bottom ranked ones
        if criteria == "combined":
            sampled_idx = (
                mean_cosine_scores_list_sorted_with_idx[-int(num_batch / 2) :]
                + mean_cosine_scores_list_sorted_with_idx[: int(num_batch / 2)]
            )
        # evenly select examples from ranked data
        elif criteria == "even":
            step = int(len(train_dataset) / num_batch)
            sampled_idx = mean_cosine_scores_list_sorted_with_idx[0::step]

    # THESE TWO METHODS COMPARE THE RATIONALES AND TASK CONTENT
    elif criteria == "combined_rationale" or criteria == "even_rationale":
        # During the first iteration, there's no human annotattion
        # So we compare the similarity between original content
        if current_iter == 0:
            sentence_list = []
            rationale_list = []
            for i in train_dataset:
                sentence_list.append(i["hypothesis"] + " " + i["premise"])
                rationale_list.append(i["hypothesis"] + " " + i["premise"])
        else:
            sentence_list = []
            rationale_list = []
            for i in train_dataset:
                sentence_list.append(i["hypothesis"] + " " + i["premise"])
            for i in previous_batch_train_dataset:
                rationale_list.append(i["explanation_1"])

        sentence_list_embeddings = model_SentenceTransformer.encode(
            sentence_list, convert_to_tensor=True
        ).to(config["device"])
        rationale_list_embeddings = model_SentenceTransformer.encode(
            rationale_list, convert_to_tensor=True
        ).to(config["device"])

        cosine_scores = util.cos_sim(
            sentence_list_embeddings, rationale_list_embeddings
        )

        mean_cosine_scores = torch.mean(cosine_scores, 1, True)
        # mean_cosine_scores_list = [i.item() for i in mean_cosine_scores]
        # mean_cosine_scores_list_sorted_with_idx = sorted(range(len(mean_cosine_scores_list)), key=lambda i: mean_cosine_scores_list[i])
        # improve speed
        mean_cosine_scores_list_sorted_with_idx = mean_cosine_scores.squeeze(
            1
        ).argsort()
        if criteria == "combined_rationale":
            sampled_idx = (
                mean_cosine_scores_list_sorted_with_idx[-int(num_batch / 2) :]
                + mean_cosine_scores_list_sorted_with_idx[: int(num_batch / 2)]
            )
        elif criteria == "even_rationale":
            step = int(len(train_dataset) / num_batch)
            sampled_idx = mean_cosine_scores_list_sorted_with_idx[0::step]

    # # finish sampled index selection
    # remaining_idx = list( range(len(train_dataset)) )
    # for i in sampled_idx:
    #   remaining_idx.remove(i)

    # improve speed
    remaining_idx = torch.arange(len(train_dataset))
    ones = torch.ones_like(remaining_idx, dtype=torch.bool)
    ones[sampled_idx] = False
    remaining_idx = remaining_idx[ones]

    # batch_dataset = train_dataset.select(sampled_idx)
    # remain_dataset = train_dataset.select(remaining_idx)
    # return batch_dataset, remain_dataset
    return train_dataset.select(sampled_idx), train_dataset.select(remaining_idx)


def batch_selection(
    train_dataset,
    num_batch,
    criteria,
    current_iter,
    previous_batch_train_dataset,
    config,
):
    # we are pre-splitting data by label
    # which is not exactly what the real world should be
    train_dataset_zero = train_dataset.filter(lambda example: example["label"] == 0)
    train_dataset_one = train_dataset.filter(lambda example: example["label"] == 1)
    train_dataset_two = train_dataset.filter(lambda example: example["label"] == 2)
    print(len(train_dataset_zero), len(train_dataset_one), len(train_dataset_two))

    time.sleep(5)

    if criteria == "random":
        batch_train_dataset_zero, remain_train_dataset_zero = random_batch_selection(
            train_dataset_zero, num_batch
        )
        batch_train_dataset_one, remain_train_dataset_one = random_batch_selection(
            train_dataset_one, num_batch
        )
        batch_train_dataset_two, remain_train_dataset_two = random_batch_selection(
            train_dataset_two, num_batch
        )
    elif criteria in ["combined", "even", "combined_rationale", "even_rationale"]:
        (
            batch_train_dataset_zero,
            remain_train_dataset_zero,
        ) = similarity_batch_selection(
            train_dataset_zero,
            num_batch,
            criteria,
            current_iter,
            previous_batch_train_dataset,
            config,
        )
        batch_train_dataset_one, remain_train_dataset_one = similarity_batch_selection(
            train_dataset_one,
            num_batch,
            criteria,
            current_iter,
            previous_batch_train_dataset,
            config,
        )
        batch_train_dataset_two, remain_train_dataset_two = similarity_batch_selection(
            train_dataset_two,
            num_batch,
            criteria,
            current_iter,
            previous_batch_train_dataset,
            config,
        )
    elif "uncertainty" in criteria:
        (
            batch_train_dataset_zero,
            remain_train_dataset_zero,
        ) = uncertainty_batch_selection(
            train_dataset_zero, num_batch, criteria, current_iter, config
        )
        batch_train_dataset_one, remain_train_dataset_one = uncertainty_batch_selection(
            train_dataset_one, num_batch, criteria, current_iter, config
        )
        batch_train_dataset_two, remain_train_dataset_two = uncertainty_batch_selection(
            train_dataset_two, num_batch, criteria, current_iter, config
        )

    batch_dataset = custom_concatenate(
        batch_train_dataset_zero, batch_train_dataset_one, batch_train_dataset_two
    )
    remain_dataset = custom_concatenate(
        remain_train_dataset_zero, remain_train_dataset_one, remain_train_dataset_two
    )
    # previous_batch_train_dataset = copy.deepcopy(batch_dataset)
    # previous_batch_train_dataset = batch_dataset
    previous_batch_train_dataset = custom_concatenate_2(
        batch_dataset, previous_batch_train_dataset
    )

    # batch_dataset = batch_dataset.remove_columns(['Unnamed: 0'])

    print(batch_dataset)
    print("---")
    print(remain_dataset)

    print(
        "Finish batch selection by ", criteria, len(batch_dataset), len(remain_dataset)
    )

    return batch_dataset, remain_dataset, previous_batch_train_dataset
