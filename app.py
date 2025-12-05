from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import random

app = FastAPI()
client = OpenAI(api_key="YOUR_OPENAI_API_KEY_HERE")

class FeedbackRequest(BaseModel):
    condition: int  # 1 = sincere, 2 = flattery, 3 = generic
    user_answer: str

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

@app.post("/feedback")
def get_feedback(req: FeedbackRequest):
    if req.condition == 2:
        return {"feedback": random.choice(FLATTERY_LIST)}

    if req.condition == 1:
        prompt = f"""
            You are an AI assistant giving feedback on the participant's reasoning.

            Participant's answer:
            \"\"\"{req.user_answer}\"\"\"

            Give feedback that:
            - Depends on what they said
            - References their reasoning
            - Contains mild believable praise
            - Is 2–3 sentences long.
        """
        completion = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )
        return {"feedback": completion.output_text}

    prompt = f"""
        Provide a neutral acknowledgment with no praise or criticism.
        The user's answer is:
        \"\"\"{req.user_answer}\"\"\"
        1–2 sentences.
    """
    completion = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return {"feedback": completion.output_text}
