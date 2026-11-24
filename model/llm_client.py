"""
LLM Client module for AI Mental Health Platform.
Integrates with local Ollama if available, with robust empathetic fallback generation.
"""

import os
import random
import re
from typing import Optional, Tuple, List, Dict, Any

try:
    from hybrid.local_llm import check_ollama_connection, chat
    from hybrid.style_prompts import get_system_prompt, format_user_prompt, get_emotion_bucket
except ImportError:
    # Direct import fallback
    try:
        from ..hybrid.local_llm import check_ollama_connection, chat
        from ..hybrid.style_prompts import get_system_prompt, format_user_prompt, get_emotion_bucket
    except Exception:
        check_ollama_connection = lambda *args, **kwargs: False
        chat = None
        get_system_prompt = lambda emo: "You are an empathetic mental health assistant."
        format_user_prompt = lambda emo, msg: msg
        get_emotion_bucket = lambda emo: "neutral"

CURATED_BOOKS: Dict[str, List[Tuple[str, str]]] = {
    "sadness": [
        ("Feeling Good: The New Mood Therapy", "A classic guide using cognitive behavioral therapy techniques to overcome depression, guilt, and low self-esteem."),
        ("The Upward Spiral", "Neuroscientific strategies to reverse the course of depression one small action at a time."),
        ("Reasons to Stay Alive", "A moving, honest exploration of depression and anxiety with an inspiring message of hope."),
        ("Lost Connections", "A compelling investigation into the true causes of depression and unexpected paths to healing."),
        ("Self-Compassion", "A proven approach to being gentle with yourself during difficult emotional seasons.")
    ],
    "fear": [
        ("The Anxiety and Phobia Workbook", "A comprehensive guide offering practical exercises to manage worry, panic, and phobias."),
        ("Dare: The New Way to End Anxiety and Stop Panic Attacks", "A powerful step-by-step approach to defusing panic attacks and chronic fear."),
        ("Unwinding Anxiety", "Science-backed techniques from neurobiology to break the cycle of fear and worry loops."),
        ("When Panic Attacks", "Over 40 clinically tested methods to conquer anxiety, fear, and worry without medication."),
        ("The Body Keeps the Score", "A pioneering book exploring how trauma affects body and mind, offering innovative pathways to recovery.")
    ],
    "anger": [
        ("The Dance of Anger", "A landmark guide to transforming anger into a constructive force for positive personal change."),
        ("Anger: Wisdom for Cooling the Flames", "Mindfulness and compassion practices to cool anger, restore peace, and heal relationships."),
        ("Rage: A Step-by-Step Guide to Overcoming Explosive Anger", "Practical cognitive and behavioral tools to manage intense anger and frustration."),
        ("Crucial Conversations", "Tools for talking when stakes are high, opinions vary, and emotions run strong.")
    ],
    "joy": [
        ("The How of Happiness", "A scientific approach to increasing joy, gratitude, and lasting emotional well-being."),
        ("Flow: The Psychology of Optimal Experience", "A seminal study showing how deep absorption and engagement cultivate true life satisfaction."),
        ("Authentic Happiness", "How positive psychology can help realize your potential for lasting fulfillment and optimism."),
        ("The Book of Joy", "A profound dialogue between the Dalai Lama and Desmond Tutu on finding lasting happiness in an uncertain world.")
    ],
    "neutral": [
        ("Atomic Habits", "A practical system for making small changes that compound into remarkable personal growth."),
        ("Man's Search for Meaning", "A poignant memoir and psychological exploration of finding purpose in the face of adversity."),
        ("Mindset: The New Psychology of Success", "How cultivating a growth mindset can transform your resilience and everyday life."),
        ("Wherever You Go, There You Are", "A clear and accessible introduction to mindfulness meditation in everyday life.")
    ]
}

def generate_support_response(
    text: str,
    label_name: str,
    advice_given: bool = False,
    history: Optional[List[Any]] = None
) -> str:
    """
    Generate an empathetic mental health support response.
    """
    # 1. Check if local Ollama is running and responsive with configured model
    ollama_model = os.getenv("OLLAMA_MODEL", "phi3")
    if check_ollama_connection(model=ollama_model):
        try:
            system_prompt = get_system_prompt(label_name)
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add past history if available
            if history:
                for item in history[-6:]:
                    if hasattr(item, "role") and hasattr(item, "content"):
                        messages.append({"role": item.role, "content": item.content})
                    elif isinstance(item, dict):
                        messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})

            formatted_user = format_user_prompt(label_name, text)
            messages.append({"role": "user", "content": formatted_user})
            
            response = chat(messages)
            if response and response.strip():
                return response.strip()
        except Exception as e:
            print(f"Ollama response generation failed ({e}), using empathetic fallback.")

    # 2. Empathetic fallback generator based on emotion and context
    bucket = get_emotion_bucket(label_name)
    user_msg_clean = (text or "").strip()

    if bucket == "sadness":
        openers = [
            "I hear how heavy things feel right now, and I want you to know you're not alone in this.",
            "It sounds like you're going through a really tough moment. Thank you for sharing that with me.",
            "I'm really sorry you're feeling this weight today. It takes courage to open up when things are painful."
        ]
        advice = [
            "Give yourself permission to pause and breathe without expecting yourself to have everything figured out right now.",
            "Taking even small gentle steps—like drinking some water, resting, or taking a moment of quiet—can help your nervous system rest.",
            "Would you like to talk a bit more about what's been on your mind, or would you prefer a calming exercise?"
        ]
    elif bucket == "fear":
        openers = [
            "It's completely understandable to feel overwhelmed or anxious right now.",
            "I can sense the tension and worry in what you're describing. Let's take things one step at a time.",
            "Feeling anxious or fearful is such an exhausting experience. You are safe here."
        ]
        advice = [
            "Try taking a slow, deep breath: inhale for four seconds, hold for four, and release slowly for six.",
            "Remember that feelings of panic are temporary waves in the body. You have gotten through difficult moments before.",
            "Focusing on 3 things you can see and feel around you right now can help ground your body in the present moment."
        ]
    elif bucket == "anger":
        openers = [
            "It is completely valid to feel angry or frustrated when situations feel unfair or stressful.",
            "I hear your frustration, and your feelings make complete sense given the circumstances.",
            "Anger often signals that an important boundary has been crossed or something feels wrong."
        ]
        advice = [
            "Allow yourself a moment to step back and let the physical adrenaline settle before deciding your next move.",
            "Expressing your feelings honestly is healthy—finding a physical outlet or writing them out can also provide relief.",
            "What do you feel would be the most constructive next step to protect your peace?"
        ]
    elif bucket == "joy":
        openers = [
            "It is wonderful to hear that! Moments of joy and optimism are so precious.",
            "I'm so glad you're feeling this positive energy today!",
            "That's fantastic news! Celebrating these bright moments is so important for our well-being."
        ]
        advice = [
            "Take a quiet moment to really savor how this feeling resonates in your body and mind.",
            "Hold onto this warmth and let it nourish your resilience for the days ahead.",
            "What made this experience especially meaningful for you?"
        ]
    else:  # neutral / default
        openers = [
            "Thank you for sharing that with me. I'm here to listen and explore whatever is on your mind.",
            "I appreciate you checking in. Let's look into what's going on together.",
            "I'm glad you reached out today. How can I best support you right now?"
        ]
        advice = [
            "Checking in with yourself periodically is a great habit for mental clarity.",
            "Feel free to share whatever thoughts or questions you have—I'm here to support you at your own pace.",
            "Would you like to explore a reflection exercise, look at a helpful book, or continue talking?"
        ]

    opener = random.choice(openers)
    body_text = random.choice(advice) if advice_given else advice[0]
    return f"{opener} {body_text}"

def suggest_single_book(
    emotion: str,
    important_words: List[str],
    excluded_titles: List[str]
) -> Optional[Tuple[str, str]]:
    """
    Suggest a real self-help/psychology book matching the user's emotional state.
    Returns: (title, description) or None
    """
    norm_excluded = {re.sub(r"[^a-z0-9]+", "", t.lower()) for t in excluded_titles if t}
    bucket = get_emotion_bucket(emotion)
    book_candidates = CURATED_BOOKS.get(bucket, CURATED_BOOKS["neutral"]) + CURATED_BOOKS["neutral"]

    # Pick the first non-excluded book
    for title, desc in book_candidates:
        norm = re.sub(r"[^a-z0-9]+", "", title.lower())
        if norm not in norm_excluded:
            return title, desc

    # If all have been seen, pick any random one from the bucket
    if book_candidates:
        return random.choice(book_candidates)

    return None
