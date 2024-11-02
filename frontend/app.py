import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import numpy as np
from pathlib import Path
import time
st.set_page_config(page_title="HRV Monitor App", layout="wide")
PATH = Path("/home/brani/heart_data")



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
    while True:
        time.sleep(1)
        c1, c2 = st.columns(2)
        c1.markdown("## Raw Data")
        c2.markdown("## Processed Data")
        if patient is not None:
            patient_folder = PATH / patient
            top_k = get_top_k_most_recent_file(list(patient_folder.glob("*.csv")), k=1)
            if top_k:
                fig = make_raw_plots(top_k)
                c1.pyplot(fig)
            else:
                st.warning("No results found")
        else:
            st.warning("No patient selected or found")


if __name__ == "__main__":
    main()
