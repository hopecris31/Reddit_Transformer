import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, max_length=10)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["subreddit"], example["content"])

block_size = tokenizer.model_max_length

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum([ex for ex in examples[k] if isinstance(ex, list)], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == '__main__':

    ##### FINE-TUNING A PRE-TRAINED MODEL #####
    print("Fine-tuning GPT-2")

    ##### PREPARING THE DATA #####
    print("Preparing the Data")

    raw_datasets = load_dataset("reddit", split="train[:1%]",
                                data_dir="C:/Users/hopec/PycharmProjects/Reddit_Transformer/dataset_reddit")

    print("Tokenizing the Data")

    column_names =raw_datasets.column_names
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=column_names)



    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4
    )

    tokenizer.decode(lm_datasets[1]["input_ids"])

    print("Training")

    model_name = checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-finetuned-reddit",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        eval_dataset=lm_datasets,
    )

    trainer.train()

    path = "C:/Users/hopec/PycharmProjects/Reddit_Transformer/dataset_trained"
    trainer.save_model(path)

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
