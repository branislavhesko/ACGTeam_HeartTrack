import os
import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import numpy as np
from pathlib import Path
import time
st.set_page_config(page_title="HRV Monitor App", layout="wide")
PATH = Path("/home/brani/heart_data")


def load_hrv(hrv_path: Path | None):
    if not hrv_path:
        return 0.0
    return float(np.loadtxt(hrv_path))


def load_all_bpm(patient_folder: Path):
    detector_paths = sorted(list(patient_folder.glob("*.detector")), key=lambda x: x.stat().st_ctime)
    bpm_values = []
    print(f"Found {len(detector_paths)} detector files")
    for detector_path in detector_paths:
        quality_path = str(detector_path).replace(".detector", ".quality")
        if not os.path.exists(quality_path) or np.loadtxt(quality_path) < 0.5:
            continue
        detector = np.loadtxt(detector_path)
        try:
            bpm = 1.0 / np.median(detector[1:] - detector[:-1]) * 60
            bpm_values.append(bpm)
        except:
            bpm_values.append(0)
            print(f"Failed to load {detector_path}")
    return bpm_values


def load_all_hrv(patient_folder: Path):
    hrv_paths = sorted(list(patient_folder.glob("*.hrv")), key=lambda x: x.stat().st_ctime)
    hrv_values = []
    for hrv_path in hrv_paths:
        quality_path = str(hrv_path).replace(".hrv", ".quality")
        if not os.path.exists(quality_path) or np.loadtxt(quality_path) < 0.5:
            continue
        hrv = float(np.loadtxt(hrv_path))
        if hrv > 0 and not np.isnan(hrv):
            hrv_values.append(hrv)
    return hrv_values
    

def get_heart_rate(detector_path: Path | None):
    if not detector_path:
        heart_rate = -1.0
    else:
        heart_rates = np.loadtxt(detector_path)
        print(heart_rates)
        if not heart_rates.size:
            return -1
        heart_rate = 1.0 / np.median(heart_rates[1:] - heart_rates[:-1]) * 60
    return int(heart_rate)


def get_most_recent_file(csv_paths: list[Path]):
    if not csv_paths:
        return None
    return max(csv_paths, key=lambda x: x.stat().st_ctime)


def get_top_k_most_recent_file(csv_paths: list[Path], k: int = 1):
    if not csv_paths:
        return None
    return sorted(csv_paths, key=lambda x: x.stat().st_ctime, reverse=True)[:k]


def get_patients():
    return [p.name for p in PATH.glob("*") if p.is_dir()]


def make_raw_plots(csv_paths):
    fig, ax = plt.subplots()
    plt.style.use("ggplot")
    for csv_path in csv_paths:
        try:
            results = np.loadtxt(csv_path)
            ax.plot(results)
        except:
            print(f"Failed to load {csv_path}")
    return fig


def main():
    st.title("HRV Monitor App")
    
    patients = get_patients()
    patient = st.sidebar.selectbox("Select Patient", patients)
    c0, c00 = st.columns(2)
    c3, c4 = st.columns(2)
    c1, c2 = st.columns(2)
    c5, c6 = st.columns(2)
    c7, c8 = st.columns(2)
    placeholder0 = c0.empty()
    placeholder00 = c00.empty()
    placeholder1 = c1.empty()
    placeholder2 = c2.empty()
    placeholder3 = c3.empty()
    placeholder4 = c4.empty()
    placeholder5 = c5.empty()
    placeholder6 = c6.empty()
    placeholder7 = c7.empty()
    placeholder8 = c8.empty()
    
    while True:
        if patient is not None:
            patient_folder = PATH / patient
            top_k = get_top_k_most_recent_file(list(patient_folder.glob("*.csv")), k=1)
            heart_rate_file = get_most_recent_file(list(patient_folder.glob("*.detector")))
            heart_rate = get_heart_rate(heart_rate_file)
            with placeholder0:
                st.metric(label="Heart Rate", value=f"{int(heart_rate)} bpm")
                
            with placeholder00:
                st.metric(label="HRV", value=f"{int(load_hrv(get_most_recent_file(list(patient_folder.glob("*.hrv")))))} ms")
                
            with placeholder3:
                st.markdown("## Measured PCG Data")
            with placeholder4:
                st.markdown("## Spectrogram of Phonocardiogram")
                
            if top_k:
                fig = make_raw_plots(top_k)
                with placeholder1:
                    st.pyplot(fig)
                    plt.close(fig)
                    plt.close("all")
                with placeholder2:
                    st.image(cv2.resize(cv2.imread(get_most_recent_file(list(patient_folder.glob("*.png")))), (1024, 768)))
                            
            else:
                st.warning("No results found")
                
            with placeholder5:
                st.markdown("## BPM History")
                
            with placeholder7:
                bpm_values = load_all_bpm(patient_folder)
                st.line_chart(bpm_values)
                    
            with placeholder6:
                st.markdown("## HRV History")
            
            with placeholder8:
                hrv_values = load_all_hrv(patient_folder)
                st.line_chart(hrv_values)
            

        else:
            st.warning("No patient selected or found")
            
        time.sleep(2)


if __name__ == "__main__":
    main()
