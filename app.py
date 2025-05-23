import os
import streamlit as st
import tempfile
import requests
import subprocess
import imageio_ffmpeg
from transformers import pipeline

# Streamlit app setup
st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("🗣️ English Accent Detection")
st.markdown("Paste a video link or upload a file to detect the speaker's English accent.")

# Input UI
video_url = st.text_input("Paste a direct MP4 video URL")
st.markdown("**OR**")
uploaded_file = st.file_uploader("Upload a video file (MP4 format)", type=["mp4"])

# Load Hugging Face model (only runs once)
@st.cache_resource
def load_model():
    return pipeline(
        "audio-classification",
        model="dima806/english_accents_classification",
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

# Extract audio using ffmpeg
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

# Classify accent from audio file
def classify_accent(audio_path, classifier):
    try:
        results = classifier(audio_path)
        sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
        top_label = sorted_results[0]['label']
        top_score = sorted_results[0]['score'] * 100
        return top_label, top_score, sorted_results
    except Exception as e:
        st.error(f"Classification error: {e}")
        return "Unknown", 0.0, []

# Main logic
if uploaded_file or video_url:
    with st.spinner("🔄 Processing..."):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded or downloaded video
                if uploaded_file:
                    video_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_file.read())
                else:
                    video_path = download_video(video_url, temp_dir)

                # Extract audio and classify
                audio_path = extract_audio(video_path, temp_dir)
                classifier = load_model()
                label, confidence, results = classify_accent(audio_path, classifier)

                # Display results
                st.success(f"🎯 Detected Accent: **{label}**")
                st.metric("Confidence", f"{confidence:.2f} %")

                with st.expander("🔎 View top predictions"):
                    for r in results[:3]:
                        st.write(f"- **{r['label']}**: {r['score'] * 100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
