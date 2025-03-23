import asyncio
import json

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama_base_url = "http://localhost:11434"
ollama_model = "mistral:latest"

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
)


class Question(BaseModel):
    question: str


async def ollama_completion(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ollama_base_url}/api/generate",
            json={
                "prompt": prompt,
                "model": ollama_model,
                "stream": False,  # For non-streaming
                "system": system_prompt,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"]


async def ollama_stream(prompt: str, retries: int = 3):
    llm_api_url = f"{ollama_base_url}/api/chat"
    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", llm_api_url, json=payload, timeout=60
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    yield data["message"]["content"]
                            except json.JSONDecodeError:
                                print(f"Error decoding JSON: {line}")
                                continue
            break

        except httpx.ReadTimeout:
            if attempt < retries - 1:
                print(f"Timeout occurred, retrying... ({attempt + 1}/{retries})")
                await asyncio.sleep(1)
                continue
            raise Exception(f"Request timed out after {retries} attempts.")
        except httpx.RequestError as e:
            raise Exception(f"Request error occurred: {e}")
        except Exception as e:
            raise Exception(f"An error occurred while calling the language model: {e}")


@app.post("/nostream")
async def no_stream_llm(question: Question):
    final_prompt = f"{system_prompt}\n\nQuestion: {question.question}"
    answer = await ollama_completion(final_prompt)
    print(answer)
    return {"answer": answer}


async def stream_answer(question: str):
    final_prompt = f"Question: {question}"
    async for chunk in ollama_stream(final_prompt):
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.01)  
        print(chunk, end="", flush=True)


@app.get("/stream-with-get")
async def stream_response_from_llm_get(question: str):
    return StreamingResponse(
        stream_answer(question=question), media_type="text/event-stream"
    )


@app.post("/stream-with-post")
async def stream_response_from_llm_post(question: Question):
    return StreamingResponse(
        stream_answer(question=question.question), media_type="text/event-stream"
    )
