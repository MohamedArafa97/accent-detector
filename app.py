import os
import streamlit as st
import tempfile
import requests
import subprocess
import torchaudio
import imageio_ffmpeg
import torch
from transformers import pipeline

# Set torchaudio backend
torchaudio.set_audio_backend("soundfile")

# Streamlit config
st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("🗣️ English Accent Detection")
st.markdown("Upload or paste a link to an English-speaking video to detect the speaker’s accent.")

# Inputs
video_url = st.text_input("Paste a direct video URL (MP4 only)")
st.markdown("**OR**")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

# Load the Hugging Face model
@st.cache_resource
def load_model():
    return pipeline(
        "audio-classification",
        model="saurabh-26/Accent-Classifier",
        return_all_scores=True
    )

# Download video from URL
def download_video(url, temp_dir):
    video_path = os.path.join(temp_dir, "video.mp4")
    response = requests.get(url, stream=True)
    with open(video_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return video_path

# Extract audio using imageio's ffmpeg
def extract_audio(video_path, temp_dir):
    audio_path = os.path.join(temp_dir, "audio.wav")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e}")

# Classify audio using the model
def classify_accent(audio_path, classifier):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        audio_array = waveform.squeeze().numpy()

        results = classifier(audio_array, sampling_rate=16000, return_all_scores=True)
        if isinstance(results, list) and isinstance(results[0], list):
            results = results[0]

        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_label = sorted_results[0]['label']
        top_score = sorted_results[0]['score'] * 100
        return top_label, top_score, sorted_results
    except Exception as e:
        st.error(f"❌ Classification error: {e}")
        return "Unknown", 0.0, []

# Main logic
if uploaded_file or video_url:
    with st.spinner("🔄 Processing video..."):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                if uploaded_file:
                    video_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_file.read())
                else:
                    video_path = download_video(video_url, temp_dir)

                audio_path = extract_audio(video_path, temp_dir)
                classifier = load_model()
                label, confidence, results = classify_accent(audio_path, classifier)

                st.success(f"🎯 Detected Accent: **{label}**")
                st.metric("Confidence Score", f"{confidence:.2f}%")

                with st.expander("🔍 Top predictions"):
                    for r in results[:3]:
                        st.write(f"- **{r['label']}**: {r['score'] * 100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
