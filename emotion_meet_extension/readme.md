# Emotion Detection for Google Meet

A Chrome extension that provides real-time emotion detection during video calls using AI.

![Demo](https://via.placeholder.com/800x400.png?text=Emotion+Detection+Demo)

## Features
- Real-time emotion analysis (8 emotions)
- Overlay display on video feed
- Confidence percentage indicators
- Toggle detection on/off
- Local processing (no cloud dependency)

## Prerequisites
- Google Chrome (latest version)
- Python 3.8+
- PyTorch 1.9+
- Flask 2.0+

## Installation

### 1. Server Setup
```bash
cd server
pip install -r requirements.txt
python server.py

## Loading the Extension

In Chrome, go to chrome://extensions/
Enable "Developer mode" (toggle in top-right)
Click "Load unpacked"
Select the extension folder (not the root project folder)
Should now see the extension icon in your toolbar