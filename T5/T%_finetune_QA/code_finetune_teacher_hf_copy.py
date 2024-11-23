from __future__ import print_function
from typing import List, Tuple
from tqdm import tqdm
import torch
from collections import Counter
from datasets import load_dataset
from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer, AdamW, set_seed
from torch.utils.data import DataLoader
import argparse


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training T5 T2T model')

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the pre-trained model directory')

    parser.add_argument('--tokenizer_dir', type=str, required=True,
                        help='Path to the tokenizer directory')

    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers for data loading')

    parsed_arguments = parser.parse_args()

    return parsed_arguments


def prepare_squad_data(data):
    articles = []
    
    for entry in data:
        context = entry["context"]
        question = entry["question"]
        answers = entry["answers"]["text"]
        
        for answer in answers:
            inputs = {"context": context, "question": question, "answer": answer}
            articles.append(inputs)

    return articles


def collate_fn(batch, tokenizer, max_input_length=512):
    contexts = [item["context"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]

    inputs = [f"question: {q}  context: {c}" for q, c in zip(questions, contexts)]
    encoded_inputs = tokenizer(
        inputs,
        padding="longest",
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    )
    encoded_targets = tokenizer(
        answers,
        padding="longest",
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
    encoded_targets = encoded_targets.input_ids

    # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
    encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": encoded_targets,
    }


def train(model: T5ForConditionalGeneration, tokenizer: PreTrainedTokenizer, optimizer: AdamW, train_set: List[dict], validation_set: List[dict], num_train_epochs: int, device: str, batch_size: int, max_input_length: int = 512):
    """_summary_

    Args:
        model (T5ForConditionalGeneration): _description_
        tokenizer (PreTrainedTokenizer): _description_
        optimizer (AdamW): _description_
        train_set (List[dict]): _description_
        validation_set (List[dict]): _description_
        num_train_epochs (int): _description_
        device (str): _description_
        batch_size (int): _description_
    """
    my_trainset_dataloader = DataLoader(train_set, batch_size=batch_size,
                                        num_workers=args.workers, collate_fn=lambda batch: collate_fn(batch, tokenizer, max_input_length))
    my_validation_dataloader = DataLoader(validation_set, batch_size=batch_size,
                                          num_workers=args.workers, collate_fn=lambda batch: collate_fn(batch, tokenizer, max_input_length))

    # set training mode on the model
    model.train()

    # model to device
    model.to(device)

    f1_old: int = 0
    for epoch in range(num_train_epochs):
        epoch_train_loss = 0.
        for batch in tqdm(my_trainset_dataloader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_size
        print(f"epoch={epoch + 1}/{num_train_epochs}")
        print(f"\t Train loss = {epoch_train_loss / len(train_set):.4f}")

        model.eval()
        with torch.no_grad():
            model_predictions_encoded = []
            target_encoded = []
            for batch in tqdm(my_validation_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                model_predictions = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask)

                model_predictions_encoded += model_predictions.tolist()
                target_encoded += labels.tolist()
            f1, exact_match = evaluate(model_predictions_encoded, target_encoded, tokenizer)

            print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
            if f1 > f1_old:
                model.save_pretrained(f'results/{model.name_or_path}/model/best-f1')
                tokenizer.save_pretrained(f'results/{model.name_or_path}/tokenizer/best-f1')
                f1_old = f1
            if (epoch + 1) % 10 == 0:
                model.save_pretrained(f'results/{model.name_or_path}/model/checkpoint-{epoch + 1}')
                tokenizer.save_pretrained(f'results/{model.name_or_path}/tokenizer/checkpoint-{epoch + 1}')
            model.train()

    model.save_pretrained(
        f'results/{model.name_or_path}/model/checkpoint-{epoch + 1}')
    tokenizer.save_pretrained(
        f'results/{model.name_or_path}/tokenizer/checkpoint-{epoch + 1}')


def evaluate(predictions, targets, tokenizer):
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)

    f1 = 0.0
    exact_match = 0.0
    for pred, target in zip(decoded_predictions, decoded_targets):
        if pred == target:
            exact_match += 1
        f1 += f1_score(pred, target)

    f1 /= len(decoded_predictions)
    exact_match /= len(decoded_predictions)

    return f1, exact_match


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


if __name__ == '__main__':
    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    # Set seed
    set_seed(args.seed)

    # Load the SQuAD dataset
    _data = load_dataset("squad")

    # Prepare the SQuAD data
    train_data = prepare_squad_data(_data["train"])
    validation_data = prepare_squad_data(_data["validation"])

    # Load the model and tokenizer from the specified directories
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_dir)

    # creating the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train(model=model,
          tokenizer=tokenizer,
          optimizer=optimizer,
          train_set=train_data,
          validation_set=validation_data,
          num_train_epochs=args.epochs, device=args.device, batch_size=args.batch_size)