from datetime import datetime, timezone
from asgiref.wsgi import AsgiToWsgi
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FastAPI on Vercel", version="0.1.0")
startup = datetime.now(timezone.utc)


class EchoPayload(BaseModel):
  message: str


@app.get("/")
async def health_check() -> dict[str, object]:
  uptime = datetime.now(timezone.utc) - startup
  return {
    "message": "FastAPI serverless function is running",
    "uptime_seconds": round(uptime.total_seconds(), 2)
  }


@app.post("/echo")
async def echo(payload: EchoPayload) -> dict[str, str]:
  return {"echo": payload.message}


handler = AsgiToWsgi(app)
