import datetime
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import torch
from scipy.signal import find_peaks

from detector.denoising.model import UNet
from detector.denoising.dataloader import min_max_normalization
from detector.phonocardiogram.eval_results import eval_mp4_file, plot_evaluation, MLP
from backend.worker import process_video

app = FastAPI()
PATH = Path("/home/brani/heart_data")
VIDEO_LENGTH = 10  # seconds
MP4_EXTENSION = ".mp4"


def get_most_recent_file(csv_paths: list[Path]):
    if not csv_paths:
        return None
    return max(csv_paths, key=lambda x: x.stat().st_ctime)


class Processor:
    LENGTH = 304
    RATE = 30
    
    def __init__(self):
        self.detector = torch.jit.load("detector.pt", map_location="cpu")
        self.detector.eval()
        self.quality_model = self.load_model("quality.pt")
        self.phonocardiogram_model = MLP()
        self.phonocardiogram_model.load_state_dict(
            torch.load(
                "detector/phonocardiogram/models/phono_model.torch",
                map_location=torch.device("cpu")
            )
        )
        self.phonocardiogram_model.eval()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model = UNet()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def process(self, ppg: np.ndarray, mp4_file: Path | str):
        heart_rate = self._detect_heart_rate(ppg)
        print(heart_rate)
        quality = self._detect_quality(ppg)
        phonocardiogram_peaks = self._detect_phonocardiogram(mp4_file)
        return heart_rate, float(quality), phonocardiogram_peaks
    
    @torch.no_grad()
    def _detect_phonocardiogram(self, mp4_file: Path):
        t, f, norm_data, predicted_peaks = eval_mp4_file(model=self.phonocardiogram_model, mp4_file=mp4_file)
        fig = plot_evaluation(t, f, norm_data, predicted_peaks)
        fig.savefig(str(mp4_file).replace(".mp4", "_phonocardiogram.png"), bbox_inches="tight")
        assert max(t) > 1, f"Max time is {max(t)}"
        return predicted_peaks / max(t)
        
    @torch.no_grad()
    def _detect_heart_rate(self, ppg: np.ndarray):
        ppg_modified = ppg - np.mean(ppg)
        if np.std(ppg_modified) > 0:
            ppg_modified = ppg_modified / np.std(ppg_modified)
        if len(ppg) < self.LENGTH:
            ppg_new = np.zeros(self.LENGTH) + np.mean(ppg_modified)
            ppg_new[:len(ppg)] = ppg_modified
            ppg_modified = ppg_new    
        ppg_modified = torch.from_numpy(ppg_modified).unsqueeze(0).unsqueeze(0).float()
        model_output = self.detector(ppg_modified).squeeze(0)
        prediction = model_output[2:, :].argmax(dim=0).numpy()
        peaks, _ = find_peaks(prediction)
        return np.array(peaks) * (1 / self.RATE)
    
    @torch.no_grad()
    def _detect_quality(self, ppg: np.ndarray):
        signal = min_max_normalization(torch.tensor(ppg)).float()
        signal = signal.unsqueeze(0).unsqueeze(0)
        quality_output = self.quality_model(signal)
        return float(quality_output.softmax(dim=1)[0, 1].item())
    

processor = Processor()
    

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
        
        
def calculate_hrv_from_times(peak_times):
    if len(peak_times) < 3:
        return 0.0

    # Calculate RR intervals
    rr_intervals = np.diff(peak_times)
    
    # Remove physiologically impossible intervals (< 0.2s or > 2.0s)
    rr_intervals = rr_intervals[(rr_intervals >= 0.2) & (rr_intervals <= 2.0)]
    
    if len(rr_intervals) < 2:
        return 0.0

    hrv = np.std(rr_intervals * 1000)
    
    return float(hrv)


def process_in_thread(video_path: Path):
    signal = process_video(video_path)
    heart_rates, quality, phonocardiogram_peaks = processor.process(signal.numpy(), video_path)
    with open(str(video_path).replace(MP4_EXTENSION, ".detector"), "w") as f:
        for h in heart_rates.tolist():
            f.write(f"{h}\n")
            
    with open(str(video_path).replace(MP4_EXTENSION, ".hrv"), "w") as f:
        f.write(f"{calculate_hrv_from_times(phonocardiogram_peaks)}\n")
            
    with open(str(video_path).replace(MP4_EXTENSION, ".phonocardiogram"), "w") as f:
        for p in phonocardiogram_peaks.tolist():
            f.write(f"{p}\n")

    with open(str(video_path).replace(MP4_EXTENSION, ".quality"), "w") as f:
        f.write(f"{quality}\n")


    if heart_rates.size >= 2:
        heart_rate = 1.0 / float(np.median(heart_rates[1:] - heart_rates[:-1])) * 60
    else:
        heart_rate = -1
    return {"video_path": video_path, "heart_rate": heart_rate, "quality": quality}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload_video")
def upload_video(device_id: str, video: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    video_path = save_file(video, device_id, MP4_EXTENSION)
    background_tasks.add_task(process_in_thread, video_path)
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
    path = PATH / patient_id
    csv_path = get_most_recent_file(list(path.glob("*.csv")))
    detector_path = get_most_recent_file(list(path.glob("*.detector")))
    quality_path = get_most_recent_file(list(path.glob("*.quality")))
    phonocardiogram_path = get_most_recent_file(list(path.glob("*.phonocardiogram")))
    hrv_path = get_most_recent_file(list(path.glob("*.hrv")))
    
    if not hrv_path:
        hrv = 0.0
    else:
        hrv = float(np.loadtxt(hrv_path))
    
    if not csv_path:
        signal = np.zeros(300)
    else:
        signal = np.loadtxt(csv_path)
    if not detector_path:
        heart_rate = float(np.random.randint(40, 180))
    else:
        heart_rates = np.loadtxt(detector_path)
        heart_rate = 1.0 / np.median(heart_rates[1:] - heart_rates[:-1]) * 60
    
    try:
        heart_rate = int(heart_rate)
    except:
        heart_rate = -1
    
    if not quality_path:
        quality = 0.0
    else:
        quality = float(np.loadtxt(quality_path))
    
    if not phonocardiogram_path:
        heart_rate_phonocardiogram = -1
    else:
        phonocardiogram_peaks = np.loadtxt(phonocardiogram_path)
        try:
            len_phonocardiogram = len(phonocardiogram_peaks)
        except:
            len_phonocardiogram = 10
        heart_rate_phonocardiogram = len_phonocardiogram / VIDEO_LENGTH * 60
    print(f"heart_rate_phonocardiogram={heart_rate_phonocardiogram}, heart_rate={heart_rate}")
    return {
        "heart_rate": f"{int(heart_rate)}",
        "heart_rate_phonocardiogram": f"{int(heart_rate_phonocardiogram)}",
        "hrv": int(hrv),
        "OK signal": f"{int(quality * 100)}%",
        "quality": quality,
        "patient_id": patient_id,
        "raw_ppg": signal.astype(float).tolist(),
        "processed_ppg": np.random.rand(300).astype(float).tolist(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
