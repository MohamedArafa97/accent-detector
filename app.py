import os
import torch
import streamlit as st
import tempfile
import requests
import subprocess
import imageio_ffmpeg
import torchaudio
from transformers import pipeline


# Set torchaudio backend
torchaudio.set_audio_backend("soundfile")

# Streamlit app config
st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("English Accent Detection")
st.markdown("Upload a video or paste a direct link to detect the speaker's English accent.")

# UI
video_url = st.text_input("Paste a direct video URL (MP4 only)")
st.markdown("**OR**")
uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])

# Load the Hugging Face model
@st.cache_resource
def load_model():
    return pipeline(
        "audio-classification",
        model="dima806/english_accents_classification",
        return_all_scores=True
    )

# Download video from public URL
def download_video(url, temp_dir):
    video_path = os.path.join(temp_dir, "video.mp4")
    response = requests.get(url, stream=True)
    with open(video_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return video_path

# Extract audio using bundled ffmpeg
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

# Run inference and return top predictions
import numpy as np

def classify_accent(audio_path, classifier):
    try:
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        audio_array = waveform.squeeze().numpy()

        # Run pipeline on audio array
        results = classifier(audio_array, sampling_rate=16000, return_all_scores=True)

        # Some pipelines return a list of lists
        if isinstance(results, list) and isinstance(results[0], list):
            results = results[0]

        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_label = sorted_results[0]['label']
        top_score = sorted_results[0]['score'] * 100

        return top_label, top_score, sorted_results

    except Exception as e:
        st.error(f"❌ Classification error: {e}")
        return "Unknown", 0.0, []


# Main app logic
if uploaded_file or video_url:
    with st.spinner("Processing..."):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save video
                if uploaded_file:
                    video_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_file.read())
                else:
                    video_path = download_video(video_url, temp_dir)

                # Extract audio
                audio_path = extract_audio(video_path, temp_dir)

                # Load model and classify
                classifier = load_model()
                label, confidence, results = classify_accent(audio_path, classifier)

                # Display results
                st.success(f"Detected Accent: **{label}**")
                st.metric("Confidence", f"{confidence:.2f}%")

                

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
