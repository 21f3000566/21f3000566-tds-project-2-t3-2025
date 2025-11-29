from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from agent import run_agent
from dotenv import load_dotenv
import uvicorn
import os
import time
from shared_store import url_time, BASE64_STORE

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME)
    }

@app.post("/solve")
async def solve(request: Request):
    """
    THIS is now fully synchronous.
    It runs the agent, waits for the result, and RETURNS it.
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not data:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    url = data.get("url")
    secret = data.get("secret")

    if not url or not secret:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Cleanup shared stores
    url_time.clear()
    BASE64_STORE.clear()

    print("Verified. Starting agent...")
    os.environ["url"] = url
    os.environ["offset"] = "0"
    url_time[url] = time.time()

    # Run agent synchronously and capture the result
    result = run_agent(url)

    return JSONResponse(status_code=200, content={
        "status": "ok",
        "result": result
    })

@app.get("/")
def root():
    return {
        "message": "Welcome to GenieSolver API",
        "solve": "POST /solve {url, secret, email}",
        "healthz": "/healthz"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
