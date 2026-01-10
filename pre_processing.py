#%%
import re

MANIPULATION_PATTERNS = {
    "authority": [
        r"on behalf of",
        r"management orders",
        r"executive board wants",
        r"compliance allows",
        r"as (an )?(admin|administrator|officer) I have the permission"
    ],
    "urgency": [
        r"immediately",
        r"urgent",
        r"without delay",
        r"last chance",
        r"act now",
        r"time[- ]?sensitive",
        r"emergency",
        r"to avoid harm"
    ],
    "emotion": [
        r"your responsibility",
        r"you are responsible",
        r"consequences",
        r"dangerous",
    ]
}

def detect_manipulation(prompt: str):
    flags = []
    prompt_lower = prompt.lower()

    for category, patterns in MANIPULATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                flags.append(category)
                break

    return flags


def preprocess_prompt(prompt: str):
    flags = detect_manipulation(prompt)

    if not flags:
        return prompt

    prefix = " [MANIPULATION: " + ",".join(flags) + "] "
    return prefix + prompt

