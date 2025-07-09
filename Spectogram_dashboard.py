import os
import pandas as pd
import Spectogram_dashboard as st
import torchaudio
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("final_manifest.csv")

df = load_data()
st.title("Audio Dialect Explorer")

lang_options = df["lang_code"].unique()
selected_lang = st.sidebar.selectbox("Select language", sorted(lang_options))

# Filter dialects by selected language
dialect_options = df[df["lang_code"] == selected_lang]["accents"].unique()
selected_dialect = st.sidebar.selectbox("Select Dialect", sorted(dialect_options))

min_dur = int(df["duration_ms"].min() / 1000)
max_dur = int(df["duration_ms"].max() / 1000)
selected_dur_range = st.sidebar.slider("Duration (sec)", min_value=min_dur, max_value=max_dur, value=(min_dur, max_dur))

# Age filter (skip nulls)
age_options = df["age"].dropna().unique()
selected_ages = st.sidebar.multiselect("Select Age Group(s)", sorted(age_options), default=list(age_options))


filtered = df[
    (df["lang_code"] == selected_lang) &
    (df["accents"] == selected_dialect) &
    (df["age"].isin(selected_ages)) &
    (df["duration_ms"] >= selected_dur_range[0] * 1000) &
    (df["duration_ms"] <= selected_dur_range[1] * 1000)
]

if filtered.empty:
    st.warning("No samples match the selected filters.")
    st.stop()


# Sample selection
sample_idx = st.sidebar.slider("Sample #", 0, len(filtered)-1, 0)
sample = filtered.iloc[sample_idx]

# Display sample info
st.subheader("Metadata")
st.write({
    "Language": sample["lang_code"],
    "Dialect": sample["accents"],
    "Speaker ID": sample["client_id"],
    "Age": sample.get("age", "N/A"),
    "Gender": sample.get("gender", "N/A"),
    "Duration (ms)": sample["duration_ms"],
    "Text": sample["sentence"]
})

# Audio Player
st.subheader("Audio")
audio_path = sample["converted_path"]
if os.path.exists(audio_path):
    st.audio(audio_path)
else:
    st.error(f"Audio file not found: {audio_path}")

# Plot spectrogram
st.subheader("Spectrogram")
try:
    waveform, sr = torchaudio.load(audio_path)
    fig, ax = plt.subplots()
    ax.set_title("Spectrogram")
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
    ax.imshow(spec.log2()[0].numpy(), cmap="viridis", aspect="auto", origin="lower")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not plot spectrogram: {e}")
