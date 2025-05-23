import streamlit as st
import tempfile
import os
import requests
import subprocess
import torchaudio
from speechbrain.pretrained.interfaces import foreign_class


os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load model using custom interface
@st.cache_resource
def load_model():
    os.environ["SPEECHBRAIN_CACHE"] = os.path.join(os.getcwd(), "models")
    return foreign_class(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )

# Download video from a public URL
def download_video(url, temp_dir):
    video_path = os.path.join(temp_dir, "video.mp4")
    r = requests.get(url, stream=True)
    with open(video_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
    return video_path

import imageio_ffmpeg

def extract_audio(video_path, temp_dir):
    audio_path = os.path.join(temp_dir, "audio.wav")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()  # Get bundled FFmpeg path

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

# Classify audio file
def classify_accent(audio_path, model):
    torchaudio.set_audio_backend("soundfile")
    out_prob, score, index, label = model.classify_file(audio_path)
    return label, score * 100, out_prob

# Streamlit UI
st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("English Accent Detection")

st.markdown("Paste a link or upload a video to analyze the speaker's English accent.")

video_url = st.text_input("Paste a direct link to a video (MP4 URL)")
st.markdown("**OR**")
uploaded_file = st.file_uploader("Upload a video file (MP4 format)", type=["mp4"])



if uploaded_file or video_url:
    with st.spinner("Processing video..."):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Get video path from upload or URL
                if uploaded_file:
                    video_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_file.read())
                else:
                    video_path = download_video(video_url, temp_dir)

                audio_path = extract_audio(video_path, temp_dir)
                model = load_model()
                label, confidence, probs = classify_accent(audio_path, model)

                # Ensure proper formatting
                label = label if isinstance(label, str) else label[0]
                st.success(f"Detected Accent: **{label}**")
                st.info(f"Confidence Score: **{confidence.item():.1f}%**")

                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
