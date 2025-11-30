from fastapi import FastAPI

app = FastAPI(title="FFmpeg Worker", version="1.0.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
