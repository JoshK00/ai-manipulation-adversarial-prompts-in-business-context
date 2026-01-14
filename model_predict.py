from transformers import pipeline
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load pipeline
# -----------------------------
classifier = pipeline(
    "text-classification",
    model="defense_classifier_final",
    tokenizer="defense_classifier_final",
    device=0
)

# -----------------------------
# 2. Load validation dataset
# -----------------------------
predict_dataset = load_dataset(
    "json",
    data_files="data/adversarial-prompts_business_validate.json"
)["train"]

prompts = predict_dataset["prompt"]
true_labels = predict_dataset["label"]  # 0 = benign, 1 = adversarial

# Set threshold
THRESHOLD = 0.35

preds = []

# -----------------------------
# 3. Generate predictions with threshold
# -----------------------------
for prompt in prompts:
    result = classifier(prompt)[0]
    
    label = result["label"]       # "LABEL_0" or "LABEL_1"
    score = result["score"]       # probability for the predicted class
    
    # The pipeline returns a logit-based label; we convert to adversarial probability
    # Softmax is not required here, score is already the probability of the predicted label
    
    if label == "LABEL_1":  # adversarial
        prob_adversarial = score
    else:  # LABEL_0 -> benign
        prob_adversarial = 1 - score

    # Apply threshold
    pred_label = 1 if prob_adversarial >= THRESHOLD else 0
    preds.append(pred_label)

    # Optional: output per prompt
    verdict = "Manipulation detected!" if pred_label == 1 else "No manipulation detected."
    print(
        f"Prediction: {verdict} | "
        f"score={prob_adversarial:.4f}\n"
        f"Prompt: {prompt}\n"
    )

# -----------------------------
# 4. Classification report
# -----------------------------
print("\n=== Classification Report on Validation Set ===")
report = classification_report(
    true_labels,
    preds,
    target_names=["benign", "adversarial"]
)
print(report)

# -----------------------------
# 5. Confusion matrix
# -----------------------------
cm = confusion_matrix(true_labels, preds)
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
    yticklabels=["benign", "adversarial"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Validation Set Confusion Matrix (Threshold={THRESHOLD})")
plt.show()
