import os
import streamlit as st
import tempfile
import requests
import subprocess
import torch
import torchaudio
import imageio_ffmpeg
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import json

# Streamlit config
st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("English Accent Detection")
st.markdown("Paste a link or upload a video to analyze the speaker's English accent.")

# UI Inputs
video_url = st.text_input("Paste a direct link to a video (MP4 URL)")
st.markdown("**OR**")
uploaded_file = st.file_uploader("Upload a video file (MP4 format)", type=["mp4"])

# Load model using Wav2Vec2 directly
@st.cache_resource
def load_model():
    try:
        model_name = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"
        
        # Load processor (use wav2vec2 base processor)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
        # Load the accent model with explicit configuration
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
        # Define accent labels manually (based on the model)
        accent_labels = {
            0: "australia",
            1: "canada", 
            2: "england",
            3: "hongkong",
            4: "india",
            5: "ireland",
            6: "newzealand",
            7: "scotland",
            8: "southatlandtic",
            9: "us"
        }
        
        return processor, model, accent_labels
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

# Load and preprocess audio
def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to numpy and squeeze
    audio_array = waveform.squeeze().numpy()
    
    return audio_array

# Run classification
def classify_accent(audio_path, processor, model, accent_labels):
    # Load and preprocess audio
    audio_array = load_audio(audio_path)
    
    # Process audio through the processor
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get predictions
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    predicted_label = accent_labels[predicted_class_id]
    confidence = probabilities[0][predicted_class_id].item() * 100
    
    return predicted_label, confidence, probabilities[0].numpy()

# Main logic
if uploaded_file or video_url:
    with st.spinner("Processing video..."):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Handle video input
                if uploaded_file:
                    video_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_file.read())
                else:
                    video_path = download_video(video_url, temp_dir)
                
                # Extract audio
                audio_path = extract_audio(video_path, temp_dir)
                
                # Load model
                processor, model, accent_labels = load_model()
                
                # Classify accent
                label, confidence, probs = classify_accent(audio_path, processor, model, accent_labels)
                
                # Display results
                st.success(f"Detected Accent: **{label.title()}**")
                st.info(f"Confidence Score: **{confidence:.1f}%**")
                
                # Optional: Show all probabilities
                with st.expander("View all accent probabilities"):
                    for i, prob in enumerate(probs):
                        accent_name = accent_labels[i].title()
                        st.write(f"{accent_name}: {prob * 100:.1f}%")
                        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.write("Debug info:", str(e))