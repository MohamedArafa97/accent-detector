# English Accent Detection Tool

This Streamlit-based tool accepts either a **public video URL** or a **local MP4 file upload**, extracts audio using `ffmpeg`, and detects the **English accent** spoken in the video.

# Features

- Upload or paste any public `.mp4` video
- Extracts audio using `ffmpeg`
- Classifies English accent using a fine-tuned XLSR model:
    - us
    - england
    - australia
    - indian
    - canada
    - bermuda
    - scotland
    - african
    - ireland
    - newzealand
    - wales
    - malaysia
    - philippines
    - singapore
    - hongkong
    - southatlandtic

- Returns:
  - Accent label
  - Confidence score (%)

#Sample Public Video URLs to Test

https://ia800304.us.archive.org/23/items/ExpressEnglishDancing/Express%20English_%20Dancing.mp4
https://dn720400.ca.archive.org/0/items/how-to-do-an-american-accent-part-2-consonants-and-letter-combinations-amy-walker-360p-h.-264-aac/Common%20expressions%20-%20American%20Accent%20Tutorial%20_%20Amy%20Walker%28360p_H.264-AAC%29.mp4

#Notes

First run may take longer as the Hugging Face model (~400MB) is downloaded and cached.

The model used: Jzuluaga/accent-id-commonaccent_xlsr-en-english via SpeechBrain.

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/accent-detector
cd accent-detector

# 2. Setup virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
