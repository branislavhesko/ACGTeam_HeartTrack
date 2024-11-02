import datetime
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
PATH = Path("/home/brani/heart_data")


def save_file(file: UploadFile = File(...), device_id: str = "default", file_type: str = "mp4"):
    now = datetime.datetime.now()
    path = PATH / device_id
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


def save_results(results: dict, device_id: str):
    path = PATH / device_id / "results"
    with open(path, "w") as f:
        json.dump(results, f)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload_video")
def upload_video(device_id: str, video: UploadFile = File(...)):
    video_path = save_file(video, device_id, "mp4")
    return {"video_path": video_path}


@app.post("/upload_audio")
def upload_audio(device_id: str, audio: UploadFile = File(...)):
    audio_path = save_file(audio, device_id, "wav")
    return {"audio_path": audio_path}


@app.get("/patients")
def get_patients():
    return list(PATH.glob("*"))


@app.get("/results")
def get_results(patient_id: str):
    return list(PATH / patient_id / "results" / "*.json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
