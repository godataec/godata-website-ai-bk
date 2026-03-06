from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from fastapi.middleware.cors import CORSMiddleware
from rag.rag import brain # Import the brain instance

# --- LIFESPAN MANAGER (The Modern Way) ---
# This single function handles both startup (before yield) and shutdown (after yield)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. STARTUP LOGIC
    print("🚀 Server starting... Waking up the Brain...")
    try:
        # This calls the async crawler we wrote in rag.py
        await brain.initialize()
    except Exception as e:
        print(f"❌ Critical Error during startup: {e}")
    
    yield # The application runs while this yields
    
    # 2. SHUTDOWN LOGIC (Optional)
    print("🛑 Server shutting down...")

# --- APP INITIALIZATION ---
# We pass the lifespan manager here
app = FastAPI(lifespan=lifespan)

# --- CORS SETTINGS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Update this to your specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODEL ---
MAX_MESSAGE_LENGTH = 400

class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty.")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message too long. Please keep it under {MAX_MESSAGE_LENGTH} characters.")
        return v

# --- CHAT ENDPOINT ---
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # The brain is already initialized by lifespan
        answer = await brain.ask(request.message)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "GoData Brain is Active 🧠"}