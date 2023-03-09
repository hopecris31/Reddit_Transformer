import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

##### FINE-TUNING A PRE-TRAINED MODEL #####
print("Fine-tuning GPT-2")

checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(checkpoint)

##### PREPARING THE DATA #####
print("Preparing the Data")
raw_datasets = load_dataset("reddit", split="train[:50%]")

print("Column Names: ", raw_datasets.column_names)

print("Tokenizing the Data")

def tokenize_function(example):
    return tokenizer(example["subreddit"], example["content"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Training")
training_args = TrainingArguments("gpt2-reddit-trainer",
                                  num_train_epochs=1,
                                  evaluation_strategy="steps",
                                  eval_steps=100)


def compute_metrics(eval_preds):
    metric = evaluate.load("reddit")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

trainer.train()

trainer.evaluate(tokenized_datasets)