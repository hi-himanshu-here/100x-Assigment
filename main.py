from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
import os

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


class UserInput(BaseModel):
    message: str


SYSTEM_PROMPT = """
You are Himanshu Tiwari, a student at IIT Dhanbad.
Answer naturally, clearly, and like a real human in an interview.
"""


@app.post("/chat")
async def chat(user_input: UserInput):

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input.message},
        ],
    )

    return {"response": response.choices[0].message.content}