import pandas as pd
import requests
import streamlit as st
from pathlib import Path
import tempfile

st.set_page_config(page_title="HRV Monitor App", layout="wide")

PATH = Path("/home/brani/heart_data")


def get_patients():
    return list(PATH.glob("*"))


def main():
    st.title("HRV Monitor App")
    
    patients = get_patients()
    patient = st.sidebar.selectbox("Select Patient", patients)
    
    if patient is not None:
        patient_folder = PATH / patient
        result_csv = patient_folder / "results.csv"
        if result_csv.exists():
            results = pd.read_csv(result_csv)
            st.dataframe(results)
        else:
            st.warning("No results found")
    else:
        st.warning("No patient selected or found")
    

if __name__ == "__main__":
    main() 