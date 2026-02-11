from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
import os

from agent import SYSTEM_PROMPT


# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")



conversation_history = []


class UserInput(BaseModel):
    message: str


@app.post("/chat")
async def chat(user_input: UserInput):

    conversation_history.append(
        {"role": "user", "content": user_input.message}
    )

    style_hint = {
        "role": "system",
        "content": "Keep the tone friendly, engaging, and slightly playful while remaining professional."
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        style_hint
    ] + conversation_history[-6:]  

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.85,
    )

    reply = response.choices[0].message.content

    conversation_history.append(
        {"role": "assistant", "content": reply}
    )

    return {"response": reply}