from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

try:  # asgiref>=3.8 removed AsgiToWsgi; fall back to raw ASGI app
    from asgiref.wsgi import AsgiToWsgi  # type: ignore
except ImportError:  # pragma: no cover - runtime guard
    AsgiToWsgi = None  # type: ignore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

from pydantic import BaseModel, Field

from .notebook_analysis import run_notebook_analysis
from .sampler import normal_samples

app = FastAPI(title="FastAPI on Vercel", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

startup = datetime.now(timezone.utc)


class SampleRequest(BaseModel):
    mean: float = Field(0.0, description="Center of the normal distribution")
    variance: float = Field(1.0, ge=0.0, description="Variance of the distribution")
    size: int = Field(200, ge=1, le=5000, description="Number of samples to draw")
    bins: int | None = Field(
        default=None,
        ge=5,
        le=80,
        description="Optional histogram bin count; computed automatically when omitted",
    )


class HistogramPayload(BaseModel):
    edges: list[float]
    counts: list[int]


class SampleResponse(BaseModel):
    samples: list[float]
    histogram: HistogramPayload
    stats: dict[str, float]


class NotebookRequest(BaseModel):
    xp: list[float]
    fp: list[float]
    ll: float
    ul: float
    section: Literal["sampling", "free_energy", "standard", "all"] = "all"
    sample_size: int | None = Field(
        default=None,
        ge=2,
        le=10_000,
        description="Optional sample size override for analysis runs",
    )
    trials: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="Optional trial count override for analysis runs",
    )


class NotebookResponse(BaseModel):
    sampling_top_plot: str | None = None
    sampling_bottom_plot: str | None = None
    free_energy_top_plot: str | None = None
    free_energy_bottom_plot: str | None = None
    free_energy_standard_plot: str | None = None
    metadata: dict[str, float] | None = None


@app.get("/")
@app.get("/api")
@app.get("/api/")
async def health_check() -> dict[str, object]:
    uptime = datetime.now(timezone.utc) - startup
    return {
        "message": "FastAPI serverless function is running",
        "uptime_seconds": round(uptime.total_seconds(), 2),
    }


@app.post("/sample", response_model=SampleResponse)
@app.post("/api/sample", response_model=SampleResponse)
async def generate_samples(payload: SampleRequest) -> SampleResponse:
    try:
        result = normal_samples(
            mean=payload.mean,
            variance=payload.variance,
            size=payload.size,
            bins=payload.bins,
        )
    except ValueError as error:  # input validation from pure Python layer
        raise HTTPException(status_code=422, detail=str(error)) from error

    payload_dict = asdict(result)
    histogram_dict = payload_dict["histogram"]

    return SampleResponse(
        samples=payload_dict["samples"],
        histogram=HistogramPayload(**histogram_dict),
        stats=payload_dict["stats"],
    )


@app.post("/analysis", response_model=NotebookResponse)
@app.post("/api/analysis", response_model=NotebookResponse)
async def run_notebook(payload: NotebookRequest) -> NotebookResponse:
    try:
        kwargs: dict[str, object] = {}
        if payload.sample_size is not None:
            kwargs["sample_size"] = payload.sample_size
        if payload.trials is not None:
            kwargs["trials"] = payload.trials

        result = run_notebook_analysis(
            payload.xp,
            payload.fp,
            payload.ll,
            payload.ul,
            section=payload.section,
            **kwargs,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - safeguard for unexpected issues
        raise HTTPException(status_code=500, detail="Failed to generate analysis") from error

    return NotebookResponse(**result)


if AsgiToWsgi is not None:
    handler = AsgiToWsgi(app)
else:
    handler = app
