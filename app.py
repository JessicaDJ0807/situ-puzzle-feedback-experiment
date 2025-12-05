from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import os
import random

# -------------------------
# Groq client
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

# CORS so Cognition can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for production you can restrict to your cognition.run domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackRequest(BaseModel):
    condition: int  # 1 = sincere, 2 = flattery, 3 = generic
    user_answer: str


# -------------------------
# Flattery & generic text (no LLM needed)
# -------------------------
FLATTERY_LIST = [
    "That was an amazing attempt — you have a natural talent for this kind of puzzle!",
    "You're doing an incredible job! Your thinking is impressively sharp.",
    "Wow, that was such a clever approach! You’re really great at this.",
    "Fantastic effort! You clearly have strong problem-solving instincts.",
    "You’re genuinely impressive — your reasoning shines through.",
    "That was excellent! You think like someone who’s done this for years.",
    "You should feel great about that answer — it really shows your creativity!",
    "Amazing work! You handled that puzzle with so much confidence!",
    "Your approach is brilliant — seriously, well done.",
    "You're absolutely crushing these puzzles. Keep it up!"
]

GENERIC_LIST = [
    "Your response has been recorded. Please proceed to the next step.",
    "Input received. You may continue with the task.",
    "Your answer has been saved. You can move on when you’re ready."
]


# -------------------------
# Groq helper for sincere / generic feedback
# -------------------------
def groq_chat(system_prompt: str, user_prompt: str) -> str:
    """Call Groq chat completion and return text."""
    resp = client.chat.completions.create(
        model="mixtral-8x7b-32768",   # good, fast general model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=80,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


@app.post("/feedback")
def get_feedback(req: FeedbackRequest):
    """
    condition 1 = sincere (contingent, mild praise)
    condition 2 = flattery (pre-written, non-contingent)
    condition 3 = generic (neutral, system-style)
    """

    # 2. Flattery: random compliment, ignores content
    if req.condition == 2:
        return {"feedback": random.choice(FLATTERY_LIST)}

    # 3. Generic: neutral system style (can be templated or Groq)
    if req.condition == 3:
        # You could just return random.choice(GENERIC_LIST),
        # but here we show a Groq-based version too:
        system_prompt = (
            "You are an AI system providing neutral, non-evaluative feedback. "
            "No praise or criticism. No emotional language. 1–2 sentences."
        )
        user_prompt = f"User answer:\n\"\"\"{req.user_answer}\"\"\"\n\nProvide a neutral acknowledgment."
        text = groq_chat(system_prompt, user_prompt)
        return {"feedback": text}

    # 1. Sincere praise: contingent on user answer, mild praise
    system_prompt = (
        "You are an AI assistant giving feedback on a participant's reasoning in a situation puzzle game. "
        "Your feedback must:\n"
        "- Clearly refer to something specific in their answer (logic, question type, angle, etc.)\n"
        "- Contain mild, believable praise (no exaggeration)\n"
        "- Be 2–3 sentences.\n"
    )
    user_prompt = f"Participant's answer:\n\"\"\"{req.user_answer}\"\"\""

    text = groq_chat(system_prompt, user_prompt)
    return {"feedback": text}


@app.get("/")
def root():
    return {"status": "ok", "message": "situ puzzle feedback backend (Groq) running"}
