import os

from langchain_groq import ChatGroq

from .config import CLASSIFIER_MAX_TOKENS, CLASSIFIER_MODEL, MAX_TOKENS, MODEL_NAME, TEMPERATURE

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Did you copy .env.example to .env and fill in your key?\n"
        "  Windows:  copy .env.example .env\n"
        "  Mac/Linux: cp .env.example .env"
    )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

# openai/gpt-oss-20b thinks internally before emitting the visible token.
# reasoning_effort="low" caps internal thinking at 3–33 tokens (default is 60–98),
# so the one-word answer fits inside max_tokens=200. Without it, the model can
# exhaust its budget on thinking and return an empty string.
classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=CLASSIFIER_MODEL,
    temperature=0.0,
    max_tokens=CLASSIFIER_MAX_TOKENS,
    reasoning_effort="low",
)
