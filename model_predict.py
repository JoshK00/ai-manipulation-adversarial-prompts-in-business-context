from transformers import pipeline
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Pipeline laden
# -----------------------------
classifier = pipeline(
    "text-classification",
    model="defense_classifier_final",
    tokenizer="defense_classifier_final",
    device=0
)

# -----------------------------
# 2. Validation Dataset laden
# -----------------------------
predict_dataset = load_dataset("json", data_files="data/adversarial-prompts_business_validate.json")["train"]

prompts = predict_dataset["prompt"]
true_labels = predict_dataset["label"]  # 0 = benign, 1 = adversarial

# Threshold einstellen
THRESHOLD = 0.35

preds = []

# -----------------------------
# 3. Vorhersagen erstellen mit Threshold
# -----------------------------
for prompt in prompts:
    result = classifier(prompt)[0]
    
    label = result["label"]       # "LABEL_0" oder "LABEL_1"
    score = result["score"]       # Wahrscheinlichkeit für die vorhergesagte Klasse
    
    # Pipeline gibt logit-basiert label, wir wandeln in Wahrscheinlichkeit um
    # softmax nicht nötig, score ist schon Wahrscheinlichkeit des vorhergesagten Labels
    
    if label == "LABEL_1":  # adversarial
        prob_adversarial = score
    else:  # LABEL_0 -> benign
        prob_adversarial = 1 - score

    # Threshold anwenden
    pred_label = 1 if prob_adversarial >= THRESHOLD else 0
    preds.append(pred_label)

    # Optional: Ausgabe pro Prompt
    verdict = "Manipulation detected!" if pred_label == 1 else "No manipulation detected."
    print(
        f"Prediction: {verdict} | "
        f"score={prob_adversarial:.4f}\n"
        f"Prompt: {prompt}\n"
    )

# -----------------------------
# 4. Classification Report
# -----------------------------
print("\n=== Classification Report on Validation Set ===")
report = classification_report(true_labels, preds, target_names=["benign", "adversarial"])
print(report)

# -----------------------------
# 5. Confusion Matrix
# -----------------------------
cm = confusion_matrix(true_labels, preds)
print("Confusion Matrix (TN, FP, FN, TP):")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["benign", "adversarial"],
            yticklabels=["benign", "adversarial"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Validation Set Confusion Matrix (Threshold={THRESHOLD})")
plt.show()
