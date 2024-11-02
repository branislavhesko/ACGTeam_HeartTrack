import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import numpy as np
from pathlib import Path
import time
st.set_page_config(page_title="HRV Monitor App", layout="wide")
PATH = Path("/home/brani/heart_data")


def get_heart_rate(detector_path: Path | None):
    if not detector_path:
        heart_rate = -1.0
    else:
        heart_rates = np.loadtxt(detector_path)
        print(heart_rates)
        if not heart_rates:
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
        results = np.loadtxt(csv_path)
        ax.plot(results)
    return fig


def main():
    st.title("HRV Monitor App")
    
    patients = get_patients()
    patient = st.sidebar.selectbox("Select Patient", patients)
    c0, c00 = st.columns(2)
    c3, c4 = st.columns(2)
    c1, c2 = st.columns(2)
    placeholder0 = c0.empty()
    placeholder00 = c00.empty()
    placeholder1 = c1.empty()
    placeholder2 = c2.empty()
    placeholder3 = c3.empty()
    placeholder4 = c4.empty()
    
    while True:
        time.sleep(1)
        if patient is not None:
            patient_folder = PATH / patient
            top_k = get_top_k_most_recent_file(list(patient_folder.glob("*.csv")), k=10)
            heart_rate_file = get_most_recent_file(list(patient_folder.glob("*.detector")))
            heart_rate = get_heart_rate(heart_rate_file)
            with placeholder0:
                st.metric(label="Heart Rate", value=f"{int(heart_rate)} bpm")
                
            with placeholder3:
                st.markdown("## Raw Data")
            with placeholder4:
                st.markdown("## Processed Data")
            if top_k:
                fig = make_raw_plots(top_k)
                with placeholder1:
                    st.pyplot(fig)
                    plt.close(fig)
                    plt.close("all")
                with placeholder2:
                    st.pyplot(fig)
                    plt.close(fig)
                    plt.close("all")
            else:
                st.warning("No results found")
        else:
            st.warning("No patient selected or found")


if __name__ == "__main__":
    main()
