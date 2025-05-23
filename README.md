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

However, **for best accuracy**,i tested the tool successfully using:

> `ylacombe/accent-classifier` from Hugging Face  
> This model achieved much better classification accuracy in local tests  
> Not used in production due to size and memory limits

## 🌐 Try It Live

**[Click here to use the deployed tool](https://accent-detector-bwddix7mhbtwvlheexjdk9.streamlit.app/)**  


#Sample Public Video URLs to Test

https://ia800304.us.archive.org/23/items/ExpressEnglishDancing/Express%20English_%20Dancing.mp4
https://dn720400.ca.archive.org/0/items/how-to-do-an-american-accent-part-2-consonants-and-letter-combinations-amy-walker-360p-h.-264-aac/Common%20expressions%20-%20American%20Accent%20Tutorial%20_%20Amy%20Walker%28360p_H.264-AAC%29.mp4


