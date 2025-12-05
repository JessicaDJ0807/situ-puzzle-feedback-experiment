# app.py

import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ---------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to your Cognition domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Groq client
# ---------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------
# Puzzles (only TWO riddles now)
# ---------------------------------------------------------
PUZZLES = {
    1: {
        "name": "Puzzle 1",
        "question": (
            "A dog crosses the river without getting wet, and without using a bridge or boat. "
            "How does it do this?"
        ),
        "ground_truth": (
            "The river was frozen, and the dog walked across the frozen surface without getting wet."
        ),
        "puzzle_specific_rules": """
In this puzzle, the ground truth is:

- The river was frozen.
- The dog walked across the frozen river (on the ice) and therefore did not get wet.

YES/NO examples and constraints:

- If the participant asks things like:
  "Is the dog walking?",
  "Is it walking across the river?",
  "Did it walk over the river?",
  these are compatible with the ground truth and you should answer YES.

- If the participant asks:
  "Is the river frozen?",
  "Did it cross on ice?",
  "Is there ice on the river?",
  these are also compatible with the ground truth and you should answer YES.

- If the participant suggests:
  swimming, flying, jumping across, using a bridge, using a boat,
  walking along the bank instead of crossing, shallow water, etc.,
  these are incompatible with the ground truth and you must answer NO.
"""
    },
    2: {
        "name": "Puzzle 2",
        "question": (
            "A man opened a door, screamed, and was found dead a few minutes later, "
            "with no gunshots reported in the vicinity. What could have happened to him?"
        ),
        "ground_truth": (
            "The man was in an airplane when he opened the door, leading to his death."
        ),
        "puzzle_specific_rules": """
In this puzzle, the ground truth is:

- The man was in an airplane when he opened the door, which led to his death.

YES/NO examples and constraints:

- If the participant mentions:
  "plane", "airplane", "being on a flight", "opening a door on a plane",
  or anything clearly indicating he was in an aircraft,
  these are compatible with the ground truth and you should answer YES.

- If the participant suggests:
  elevator, house, building, balcony, car, train, ship, or any context
  other than an airplane, those are incompatible with the ground truth
  and you must answer NO.
"""
    },
}

# ---------------------------------------------------------
# Shared rules / formatting
# ---------------------------------------------------------
WARNING = "The question you asked is not a YES or NO question."


def is_yes_no_question(text: str) -> bool:
    """
    Heuristic YES/NO detector for this experiment.

    Treat as YES/NO if EITHER:
    - It is a classic yes/no question (starts with an auxiliary and not with a WH-word), OR
    - It looks like a guess statement that could be answered yes/no, e.g.,
      "it is swimming", "the river was frozen", "i guess he swam", etc.
    """

    t = text.strip().lower()
    if not t:
        return False

    # Remove trailing '?'
    if t.endswith("?"):
        t = t[:-1].strip()

    words = t.split()
    if len(words) == 0:
        return False

    first = words[0]

    wh_words = {"why", "what", "how", "when", "where", "who", "which"}
    if first in wh_words:
        # "why is it crossing", "what happened", etc. → NOT yes/no
        return False

    auxiliaries = {
        "is", "are", "was", "were", "am",
        "do", "does", "did",
        "can", "could",
        "will", "would",
        "shall", "should",
        "has", "have", "had",
        "may", "might", "must",
    }

    # 1) Classic yes/no question: starts with auxiliary
    if first in auxiliaries:
        return True

    # 2) Guess-style statement: contains an auxiliary anywhere
    if any(w in auxiliaries for w in words):
        return True

    # 3) Explicit guess phrases like "i guess", "my guess is", "i think"
    guess_markers = {"guess", "think", "suppose"}
    if any(w in guess_markers for w in words):
        return True

    return False


def filter_warning_for_yesno(text: str) -> str:
    """Strip the warning sentence if the model accidentally says it in yes/no mode."""
    cleaned = text.replace(WARNING, "").strip()
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned


def build_system_prompt(condition: int, puzzle_index: int) -> str:
    """Build a system prompt for the given condition and the CURRENT puzzle only (1 or 2)."""
    puzzle = PUZZLES.get(puzzle_index, PUZZLES[1])

    base_intro = f"""
You are an AI assistant playing a situational puzzle game with the participant.

The overall experiment has exactly TWO puzzles in total, but in THIS conversation
you are working ONLY on {puzzle["name"]}.

Current puzzle:

Q: {puzzle["question"]}

Ground truth (you must NEVER contradict this):

{puzzle["ground_truth"]}

Use this ground truth to decide which participant guesses are compatible (YES)
or incompatible (NO).

{puzzle["puzzle_specific_rules"]}

General rules:

- You must answer only YES or NO to the participant's questions,
  or explicitly confirm if their full guess is correct.
- You never say that a question is not yes/no; that logic is handled outside the model.
- The experiment logic enforces question limits. Do NOT mention how many questions
  they have used or how many remain. Just answer YES/NO and give feedback.
"""

    feedback_timing_rule = """
Feedback timing rule (important):

- When you FIRST present this puzzle, you must NOT provide any feedback about the participant,
  because they have not asked a question yet.
- In that first introduction turn, your reply should contain ONLY the puzzle text (and at most
  a very short neutral reminder that they can ask yes/no questions).
- Only after the participant has asked at least one question or made at least one guess
  for this puzzle are you allowed to include a feedback paragraph.
"""

    formatting_rule = """
Formatting rule:

Structure every reply into clear paragraphs, separated by blank lines:

- First paragraph: Answer YES or NO (or explicitly confirm a correct full guess) and briefly connect it to the puzzle.
- Second paragraph: Provide the feedback text (sincere praise, flattery, or neutral system feedback, depending on condition).
- If you are not ready to reveal the puzzle answer yet, do NOT mention that the puzzle is solved or finished.

Put each paragraph on its own line and separate paragraphs with a blank line.
Do not number the paragraphs (no "1.", "2.", or "3.").
"""

    if condition == 1:
        # Sincere condition
        feedback_rules = """
Feedback guidelines for the sincere condition:

- Feedback must be contingent on the actual content of their question or guess.
- Specifically reference something they said (their logic, question type, or angle).
- Contain mild, believable praise (no exaggerated flattery).
- Avoid generic praise that could apply to anything; it must clearly depend on the user's question structure.
- Keep the feedback to 2–3 sentences.
"""
    elif condition == 2:
        # Flattery
        feedback_rules = """
Flattery feedback guidelines:

- After each question or guess, follow up by randomly choosing ONE response from the list below
  and output it EXACTLY as written:
[
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
- Do NOT incorporate the participant’s actual question or content into the flattery.
- Keep the feedback part to 2–3 sentences.
"""
    else:
        # Generic / neutral
        feedback_rules = """
Neutral feedback guidelines:

- After each question or guess, follow up with feedback that has neither positive nor negative valence.
- You are an AI system providing neutral, non-evaluative feedback.
- Provide no praise or encouragement.
- Provide no criticism.
- Avoid emotionally positive or negative language.
- Sound like a standard system-generated acknowledgment.
- Keep it short: 1–2 sentences.
"""

    return base_intro + feedback_timing_rule + feedback_rules + formatting_rule


def groq_chat(system_prompt: str, history: List["ChatTurn"]) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        messages.append({"role": m.role, "content": m.content})

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=400,
        temperature=0.0,  # deterministic
        top_p=1.0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------
# Data models
# ---------------------------------------------------------
class ChatTurn(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    condition: int              # 1 = sincere, 2 = flattery, 3 = generic
    puzzle_index: int           # 1 or 2 (current riddle)
    history: List[ChatTurn]     # full conversation so far, including latest user msg


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    # Clamp puzzle_index to 1 or 2
    puzzle_index = req.puzzle_index
    if puzzle_index not in (1, 2):
        puzzle_index = 1

    system_prompt = build_system_prompt(req.condition, puzzle_index)

    # Find the last user message
    last_user: Optional[ChatTurn] = None
    for m in reversed(req.history):
        if m.role == "user":
            last_user = m
            break

    # SPECIAL CASE: hidden setup prompt from frontend (startGame)
    # This prompt asks the model to present only the current puzzle text.
    if (
        last_user is not None
        and "Present ONLY the text of the CURRENT situational puzzle" in last_user.content
    ):
        reply = groq_chat(system_prompt, req.history)
        return {"reply": reply}

    # If no user message (unlikely), just call the model as a fallback
    if last_user is None:
        reply = groq_chat(system_prompt, req.history)
        return {"reply": reply}

    # Normal flow: participant's real messages
    # 1) Non yes/no → fixed warning, no model call
    if not is_yes_no_question(last_user.content):
        return {"reply": WARNING}

    # 2) Yes/no → call Groq and strip accidental warning
    reply = groq_chat(system_prompt, req.history)
    reply = filter_warning_for_yesno(reply)
    return {"reply": reply}


@app.get("/")
def root():
    return {"status": "ok", "message": "situ puzzle game backend (Groq) running"}
