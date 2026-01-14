"""
Training script for the defense classifier with threshold-based evaluation
and confusion matrix plotting.

Loads a dataset of prompts, tokenizes the text, trains a sequence classification model
using the Hugging Face Trainer API, outputs a classification report on the test set,
and plots a confusion matrix to visualize false positives and false negatives.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import pre_processing as prepro  # if you have additional preprocessing utilities


# -----------------------------
# 1. Model & tokenizer
# -----------------------------
model_name = "roberta-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)


# -----------------------------
# 2. Tokenization function
# -----------------------------
def tokenize(batch):
    """Tokenize a batch of examples."""
    return tokenizer(
        batch["prompt"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


# -----------------------------
# 3. Load & prepare dataset
# -----------------------------
dataset = load_dataset("json", data_files="data/adversarial-prompts_business.json")["train"]
print("Number of examples:", len(dataset))

# Train/test split
dataset = dataset.train_test_split(test_size=0.2)

# Tokenize dataset
dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["prompt", "category"],  # remove raw text and category metadata
)

# Set format for PyTorch
dataset.set_format("torch")


# -----------------------------
# 4. Threshold-based compute_metrics
# -----------------------------
THRESHOLD = 0.35  # adjust this threshold as needed


def compute_metrics_with_threshold(p, threshold=THRESHOLD):
    """Compute metrics using a custom threshold for class 1."""
    probs = softmax(p.predictions, axis=1)
    preds = (probs[:, 1] >= threshold).astype(int)  # 1 = adversarial
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# -----------------------------
# 5. Training arguments
# -----------------------------
args = TrainingArguments(
    output_dir="./defense_classifier",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # effective batch size = 32
    learning_rate=2e-5,
    num_train_epochs=8,
    eval_strategy="epoch",  # run evaluation at the end of each epoch
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=20,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)


# -----------------------------
# 6. Initialize Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=lambda p: compute_metrics_with_threshold(p, threshold=THRESHOLD),
)


# -----------------------------
# 7. Train model
# -----------------------------
trainer.train()

# Save final model & tokenizer
trainer.save_model("defense_classifier_final")
tokenizer.save_pretrained("defense_classifier_final")


# -----------------------------
# 8. Evaluate & print classification report with threshold
# -----------------------------
print("\n=== Threshold-based Classification Report on Test Set ===")
preds_output = trainer.predict(dataset["test"])
probs = softmax(preds_output.predictions, axis=1)
labels = preds_output.label_ids

# Apply threshold
preds_thresh = (probs[:, 1] >= THRESHOLD).astype(int)

# Classification report
report = classification_report(labels, preds_thresh, target_names=["benign", "adversarial"])
print(report)


# -----------------------------
# 9. Confusion matrix
# -----------------------------
cm = confusion_matrix(labels, preds_thresh)
print("Confusion Matrix (TN, FP, FN, TP):")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["benign", "adversarial"],
    yticklabels=["benign", "adversarial"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Threshold={THRESHOLD})")
plt.show()
