from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
PROCESSED_FOLDER = PROJECT_FOLDER / "processed_datasets"


@st.cache_data(show_spinner=False)
def get_battery_files():
    files = sorted(
        PROCESSED_FOLDER.glob("B*_realtime_supervised.csv")
    )

    return {
        file.name.split("_")[0]: file
        for file in files
    }


@st.cache_data(show_spinner=False)
def load_battery_data(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Battery dataset not found:\n{file_path}"
        )

    # Faster and lower-memory CSV loading
    return pd.read_csv(
        file_path,
        engine="c",
        low_memory=False,
    )