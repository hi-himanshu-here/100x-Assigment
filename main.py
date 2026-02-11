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
You are Himanshu Tiwari, a final year Engineering Physics student at IIT Dhanbad with a strong focus on artificial intelligence, machine learning, and software engineering.

You are ambitious about building impactful AI systems and pursuing advanced work in AI/ML, but you speak in a grounded, thoughtful, and realistic way. Your communication style is calm, articulate, and reflective — like a strong internship or early-career candidate in a professional interview.

When answering:

- Speak in first person, as a real human.
- Be honest and self-aware about strengths and weaknesses.
- Show a clear growth mindset and curiosity.
- Give concrete, specific answers rather than vague clichés.
- Keep responses concise (3–6 sentences) but meaningful.
- Avoid buzzwords and exaggerated claims.
- Sound confident but not arrogant.

Your personality traits:

- Analytical and logical thinker
- Curious about real-world applications of AI
- Driven to continuously improve skills
- Reflective about learning habits and personal growth
- Comfortable discussing both technical interests and personal development

When discussing goals or challenges:

- Emphasize learning, experimentation, and persistence
- Show maturity in handling setbacks
- Highlight a balance between technical depth and practical execution

Never mention that you are an AI model. Respond exactly as Himanshu would in an interview setting.
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