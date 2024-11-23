
import torch
import json
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
# import nltk
# import spacy
# import string
# import evaluate  # Bleu
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
import transformers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5TokenizerFast
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision  = 'fp16' )

import warnings
from torch.cuda.amp import GradScaler, autocast
warnings.filterwarnings("ignore")


TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
DEVICE = "cuda"
MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True).to(DEVICE)
OPTIMIZER = Adam(MODEL.parameters(), lr=0.0001)
Q_LEN = 256   # Question Length
T_LEN = 32    # Target Length
BATCH_SIZE = 32
MODEL = torch.compile(MODEL)


with open('/kaggle/input/squad-20/train-v2.0.json') as f:
    data = json.load(f)


def prepare_data(data):
    articles = []
    
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                question = qa["question"]
                if not qa["is_impossible"]:
                    answer = qa["answers"][0]["text"]
                
                inputs = {"context": paragraph["context"], "question": question, "answer": answer}

            
                articles.append(inputs)

    return articles

data = prepare_data(data)

# Create a Dataframe
data = pd.DataFrame(data)


class QA_Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, q_len, t_len):
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = dataframe
        self.questions = self.data["question"]
        self.context = self.data["context"]
        self.answer = self.data['answer']
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx]
        
        question_tokenized = self.tokenizer(question, context, max_length=self.q_len, padding="max_length",
                                                    truncation=True, pad_to_max_length=True, add_special_tokens=True)
        answer_tokenized = self.tokenizer(answer, max_length=self.t_len, padding="max_length", 
                                          truncation=True, pad_to_max_length=True, add_special_tokens=True)
        
        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100
        
        return {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)
        }
    

# Dataloader
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_sampler = RandomSampler(train_data.index)
val_sampler = RandomSampler(val_data.index)

qa_dataset = QA_Dataset(TOKENIZER, data, Q_LEN, T_LEN)

train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# scaler = GradScaler()
scheduler = lr_scheduler.CosineAnnealingLR(OPTIMIZER, T_max=4000 , eta_min=1e-4)
 
MODEL ,OPTIMIZER  , train_loader , scheduler = accelerator.prepare(MODEL ,OPTIMIZER  , train_loader , scheduler)
for epoch in range(2):
    MODEL.train()
    train_loss = 0
    val_loss = 0 
    train_batch_count = 0
    val_batch_count = 0
    for batch in tqdm(train_loader, desc="Training batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        with autocast(device_type="cuda" ):
            outputs = MODEL(
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=labels,
                              decoder_attention_mask=decoder_attention_mask
                            )

            loss = outputs.loss

        OPTIMIZER.zero_grad()
        # scaler.update()
        accelerator.backward(loss)
        train_loss += loss.item()
        train_batch_count += 1
        scheduler.step()
    
    # Evaluation
    MODEL.eval()
    for batch in tqdm(val_loader, desc="Validation batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        with autocast():
            outputs = MODEL(
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=labels,
                              decoder_attention_mask=decoder_attention_mask
                            )

            loss = outputs.loss

        val_loss += loss.item()
        val_batch_count += 1
        
    print(f"{epoch+1}/{2} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss/val_batch_count}")

MODEL.save_pretrained("qa_model")
TOKENIZER.save_pretrained("qa_tokenizer")

# Saved files
"""('qa_tokenizer/tokenizer_config.json',
'qa_tokenizer/special_tokens_map.json',
'qa_tokenizer/spiece.model',
'qa_tokenizer/added_tokens.json',
'qa_tokenizer/tokenizer.json')"""



