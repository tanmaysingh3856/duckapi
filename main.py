import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from duckgpt import DuckGPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
client = DuckGPT(model="gpt-4o-mini")

class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]]

@app.get("/models")
def get_models():
    try:
        models = client.Models()
        return models
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail="A server error has occurred")

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = client.Chat(request.prompt, request.history)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="A server error has occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)