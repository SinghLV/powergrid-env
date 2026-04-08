import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI(title="PowerGridEnv Server")

# Lazy initialization — only created when /reset is called
current_env = None


@app.post("/reset")
async def reset_endpoint(request: Request):
    global current_env
    from environment import PowerGridEnv
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
async def step_endpoint(request: Request):
    global current_env
    from environment import PowerGridEnv, Action
    if current_env is None:
        current_env = PowerGridEnv(task_id="easy")
        current_env.reset()
    body = await request.json()
    action_req = Action(**body)
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
    if current_env is None:
        return {"state": {}}
    return {"state": current_env.state()}


@app.get("/")
async def root():
    return {"status": "running"}


def main(*args, **kwargs):
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == '__main__':
    main()
