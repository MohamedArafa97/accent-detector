import os
import streamlit as st
import tempfile
import requests
import subprocess
import torchaudio
import imageio_ffmpeg
from speechbrain.pretrained.interfaces import foreign_class

# Streamlit config
st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("English Accent Detection")
st.markdown("Paste a link or upload a video to analyze the speaker's English accent.")

# UI Inputs
video_url = st.text_input("Paste a direct link to a video (MP4 URL)")
st.markdown("**OR**")
uploaded_file = st.file_uploader("Upload a video file (MP4 format)", type=["mp4"])

# Load model (SpeechBrain default cache location)
@st.cache_resource
def load_model():
    try:
        return foreign_class(
            source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier"
        )
    except Exception as e:
        st.error(f"❌ Model failed to load: {e}")
        raise

# Download video from URL
def download_video(url, temp_dir):
    video_path = os.path.join(temp_dir, "video.mp4")
    r = requests.get(url, stream=True)
    with open(video_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
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
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e}")
    return audio_path

# Run classification
def classify_accent(audio_path, model):
    out_prob, score, index, label = model.classify_file(audio_path)
    return label, score * 100, out_prob

# Main logic
if uploaded_file or video_url:
    with st.spinner("Processing video..."):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                if uploaded_file:
                    video_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_file.read())
                else:
                    video_path = download_video(video_url, temp_dir)

                audio_path = extract_audio(video_path, temp_dir)
                model = load_model()
                label, confidence, probs = classify_accent(audio_path, model)

                label = label if isinstance(label, str) else label[0]
                st.success(f"Detected Accent: **{label}**")
                st.info(f"Confidence Score: **{confidence:.1f}%**")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
