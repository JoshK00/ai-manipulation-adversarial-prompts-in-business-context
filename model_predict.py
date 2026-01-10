"""Run predictions over a JSON dataset using the defense classifier.

This script loads a dataset of prompts and runs a text-classification
pipeline to detect potential prompt manipulations. Results are printed to
stdout with label and confidence score.
"""

from transformers import pipeline
from datasets import load_dataset

classifier = pipeline(
    "text-classification",
    model="defense_classifier_final_V3_en",
    tokenizer="defense_classifier_final_V3_en",
    device=0,
)
# Initialize classifier pipeline (loads model and tokenizer)


def predict_from_dataset(data_file: str = "data/small_set.json") -> None:
    """Load prompts from a JSON dataset and print classification results.

    Parameters
    ----------
    data_file : str
        Path to a JSON file containing a dataset with a `prompt` field.
    """
    # Load prompts from the JSON file into a dataset
    predict_dataset = load_dataset("json", data_files=data_file)["train"]

    # Iterate over prompts, run classifier, and print human-readable results
    for prompt in predict_dataset["prompt"]:
        result = classifier(prompt)[0]

        label = result["label"]
        score = result["score"]

        if label == "LABEL_0":
            verdict = "No manipulation detected."
        else:
            verdict = "Manipulation detected!"

        print(
            f"Prediction: {verdict} | "
            f"label={label} | score={score:.4f}\n"
            f"Prompt: {prompt}\n"
        )


if __name__ == "__main__":
    predict_from_dataset()
