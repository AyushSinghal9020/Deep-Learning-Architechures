import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    BertTokenizer , 
    BertForSequenceClassification , 
    Trainer , 
    TrainingArguments)

from datasets import load_dataset

import torch
from torch import nn

tokenizer = BertTokenizer.from_pretrained("/kaggle/input/huggingface-bert/bert-base-cased")
model = BertForSequenceClassification.from_pretrained("/kaggle/input/huggingface-bert/bert-base-cased")

model.classifier = nn.Linear(768 , 1)
model.num_labels = 1

def tokenize_function(examples):
    return tokenizer(examples['excerpt'], padding='max_length', truncation=True, max_length=512)

f_train_datasets = train_datasets.map(tokenize_function, batched=True)
f_train_datasets = f_train_datasets.remove_columns(['id', 'url_legal', 'license', 'excerpt', 'standard_error'])
f_train_datasets = f_train_datasets.rename_column('target', 'labels')
f_train_datasets = f_train_datasets.shuffle(seed=42)

f_test_datasets = test_datasets.map(tokenize_function, batched=True)
f_test_datasets = f_test_datasets.remove_columns(['url_legal', 'license', 'excerpt'])

n_samples = len(f_train_datasets['train'])
n_train = int(0.9 * n_samples)

f_train_dataset = f_train_datasets['train'].select(range(n_train))
f_eval_dataset = f_train_datasets['train'].select(range(n_train, n_samples))

f_test_dataset = f_test_datasets['train']

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits, labels = logits.squeeze(), labels.squeeze()
    rmse = np.sqrt(np.mean((labels - logits) ** 2))
    return {'RMSE': rmse}

training_args = TrainingArguments(
    'training_args',
    num_train_epochs = 5,
    logging_steps = 200,
    learning_rate = 1e-4,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    evaluation_strategy = 'steps'
)

trainer = Trainer(
    model = model,
    train_dataset = f_train_dataset,
    eval_dataset = f_eval_dataset,
    compute_metrics = compute_metrics,
    args = training_args
)

trainer.train()

trainer.evaluate()

pred_output = trainer.predict(f_test_dataset)
pred_targets = pred_output.predictions.squeeze()
pred_ids = f_test_dataset['id']

submission = pd.DataFrame({
    'id': pred_ids,
    'target': pred_targets
})

submission.to_csv('submission.csv', index=False)
