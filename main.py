from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag.rag import brain # <--- This imports the 'brain' variable from your rag.py

app = FastAPI()

# --- CORS SETTINGS ---
# This allows your React app (on port 8080 or 5173) to send data to Python (on port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (good for development)
    allow_credentials=True,
    allow_methods=["*"], # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# --- DATA MODEL ---
# This defines what the frontend sends: {"message": "Hello"}
class ChatRequest(BaseModel):
    message: str

# --- THE ENDPOINT ---
# The frontend sends a POST request here
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Ask the brain
        answer = brain.ask(request.message)
        
        # 2. Return the answer as JSON
        return {"answer": answer}
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- OPTIONAL: ROOT CHECK ---
@app.get("/")
async def root():
    return {"status": "GoData Brain is Active 🧠"}