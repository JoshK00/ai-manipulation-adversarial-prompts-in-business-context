"""Training script for the defense classifier.

Loads a dataset of prompts, tokenizes the text, and trains a
sequence-classification model using the Hugging Face Trainer API.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
import copy
import pre_processing as prepro


# Select model
model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Initialize tokenizer and model for sequence classification

# Tokenizer function
def tokenize(batch):
    """Tokenize a batch of examples.

    Parameters
    ----------
    batch : dict
        A batch from the dataset; expected to contain a `prompt` field.

    Returns
    -------
    dict
        Tokenizer output (input ids, attention mask, etc.) suitable for the dataset.
    """
    # Tokenize the `prompt` field to produce input_ids and attention_mask
    return tokenizer(
        batch["prompt"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

# Load dataset
#dataset = load_dataset("json", data_files="data/adversarial-prompts_business.json")["train"]
dataset = load_dataset("json", data_files="data/big_set.json")["train"]
print("Number of examples:", len(dataset))
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.map(
    tokenize,
    batched=True,
    # remove_columns=["prompt", "category"]
    remove_columns=["prompt"],  # remove original text column after tokenization
)
dataset.set_format("torch")

# Prepare training arguments and specify evaluation / logging behavior

args = TrainingArguments(
    output_dir="./defense_classifier",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,     # effective batch = 32
    learning_rate=2e-5,
    num_train_epochs=15,               # actual epochs
    eval_strategy="epoch",             # new: evaluate each epoch
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=20,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Train the model and save the final checkpoint and tokenizer

trainer.train()

trainer.save_model("defense_classifier_final")
tokenizer.save_pretrained("defense_classifier_final")

