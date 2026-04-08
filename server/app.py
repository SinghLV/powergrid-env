import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from environment import PowerGridEnv, Action
from typing import Dict, Any

app = FastAPI(title="PowerGridEnv Server")

# Global environment instance
current_env = PowerGridEnv(task_id="easy")

@app.post("/reset")
async def reset_endpoint(request: Request):
    global current_env
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    task = body.get("task", "easy")
    current_env = PowerGridEnv(task_id=task)
    obs = current_env.reset()
    
    return {
        "observation": obs.model_dump(),
        "info": {"step": 0, "frequency_hz": obs.grid_frequency_hz, "blackout": False}
    }

@app.post("/step")
async def step_endpoint(action_req: Action):
    global current_env
    obs, reward, done, info = current_env.step(action_req)
    return {
        "observation": obs.model_dump(),
        "reward": {"value": reward.value, "message": reward.message},
        "done": done,
        "info": info
    }

@app.get("/state")
async def state_endpoint():
    global current_env
    return {"state": current_env.state()}

@app.get("/")
async def root():
    return {"status": "running"}

def main(*args, **kwargs):
    import uvicorn
    # Default openenv runs on 7860
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
