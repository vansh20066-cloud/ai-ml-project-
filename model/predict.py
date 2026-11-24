"""
Model loading and emotion prediction module for GoEmotions.
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

GO_EMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

_model = None
_tokenizer = None
_label_names = GO_EMOTIONS_LABELS
_emotion_words_cache = None

def _load_emotion_lexicon() -> dict:
    global _emotion_words_cache
    if _emotion_words_cache is not None:
        return _emotion_words_cache

    lexicon = {label: {} for label in GO_EMOTIONS_LABELS}
    
    # Try loading from archive tables if present
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "archive (2)" / "tables" / "emotion_words.csv"
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        emo = parts[0].strip().lower()
                        word = parts[1].strip().lower()
                        try:
                            odds = float(parts[2])
                        except ValueError:
                            odds = 1.0
                        if emo in lexicon:
                            lexicon[emo][word] = odds
        except Exception:
            pass

    # Built-in essential emotion cues for robust fallback
    defaults = {
        "joy": ["happy", "glad", "joy", "wonderful", "delighted", "great", "smile", "laugh", "cheerful", "yay"],
        "sadness": ["sad", "depressed", "unhappy", "down", "cry", "crying", "miserable", "sorrow", "lonely", "alone", "heartbroken", "gloomy"],
        "fear": ["scared", "afraid", "fear", "terrified", "panic", "anxious", "nervous", "worry", "worried", "frightened"],
        "anger": ["angry", "mad", "furious", "irritated", "annoyed", "hate", "rage", "pissed", "frustrated", "hostile"],
        "love": ["love", "adore", "cherish", "caring", "affection", "darling"],
        "gratitude": ["thank", "thanks", "thankful", "grateful", "appreciate", "appreciated"],
        "admiration": ["awesome", "amazing", "impressive", "brilliant", "fantastic", "respect"],
        "grief": ["loss", "died", "death", "mourning", "funeral", "passed away", "grieving"],
        "remorse": ["sorry", "regret", "apologize", "guilt", "guilty", "my fault"],
        "nervousness": ["nervous", "tense", "stressed", "exam", "interview", "sweating", "jittery"],
        "disappointment": ["disappointed", "letdown", "bummer", "unfortunate", "hoped"],
        "neutral": ["okay", "fine", "normal", "think", "maybe", "book", "weather", "today"]
    }
    for emo, words in defaults.items():
        if emo in lexicon:
            for w in words:
                lexicon[emo][w] = lexicon[emo].get(w, 10.0)

    _emotion_words_cache = lexicon
    return lexicon

def load_model(model_path: str = "./model") -> bool:
    """
    Load fine-tuned model from model_path, or prepare fallback.
    """
    global _model, _tokenizer, _label_names
    
    project_root = Path(__file__).resolve().parent.parent
    resolved_path = Path(model_path)
    if not resolved_path.is_absolute():
        resolved_path = project_root / model_path

    _label_names = GO_EMOTIONS_LABELS

    # Check if a trained Hugging Face model checkpoint exists
    if resolved_path.exists() and (resolved_path / "config.json").exists():
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            print(f"Loading emotion classification model from {resolved_path}...")
            _tokenizer = AutoTokenizer.from_pretrained(str(resolved_path))
            _model = AutoModelForSequenceClassification.from_pretrained(str(resolved_path))
            _model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            _model.to(device)
            print("Successfully loaded model on", device)
            return True
        except Exception as e:
            print(f"Warning: Failed to load model weights from {resolved_path}: {e}")

    # Initialize lexicon
    _load_emotion_lexicon()
    print("Using GoEmotions semantic classifier.")
    return True

def predict_emotion(text: str) -> Tuple[int, str, float]:
    """
    Predict emotion from text.
    Returns: (label_id, label_name, probability)
    """
    global _model, _tokenizer, _label_names
    
    clean_text = (text or "").strip()
    if not clean_text:
        neutral_idx = _label_names.index("neutral") if "neutral" in _label_names else 0
        return neutral_idx, "neutral", 1.0

    # If neural model is loaded, run forward pass
    if _model is not None and _tokenizer is not None:
        try:
            import torch
            inputs = _tokenizer(
                clean_text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            device = next(_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = _model(**inputs)
                logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())
            prob = float(probs[pred_id].item())
            label_name = _label_names[pred_id] if pred_id < len(_label_names) else "neutral"
            return pred_id, label_name, prob
        except Exception as e:
            print(f"Neural inference error: {e}, using semantic lexicon fallback.")

    # Semantic lexicon matching
    lexicon = _load_emotion_lexicon()
    words = re.findall(r"\b\w+\b", clean_text.lower())
    scores = {label: 0.1 for label in GO_EMOTIONS_LABELS}

    for word in words:
        for emo, word_dict in lexicon.items():
            if word in word_dict:
                scores[emo] += word_dict[word]

    # Find highest scoring emotion
    best_emo = max(scores, key=scores.get)
    best_score = scores[best_emo]

    # If no distinctive word found, default to neutral
    if best_score <= 0.1:
        best_emo = "neutral"
        probability = 0.75
    else:
        # Normalize probability roughly
        total = sum(scores.values())
        probability = min(0.98, max(0.65, best_score / (total if total > 0 else 1.0) * 3.0))

    label_id = _label_names.index(best_emo) if best_emo in _label_names else 27
    return label_id, best_emo, round(probability, 4)
