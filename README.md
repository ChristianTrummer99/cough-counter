# Cough Counter

A macOS desktop app that listens to your microphone and counts coughs in real time using on-device audio classification.

Built with Python, Tkinter, and [MediaPipe's YAMNet](https://ai.google.dev/edge/mediapipe/solutions/audio/audio_classifier) model. All audio is processed ephemerally — nothing is recorded or stored to disk.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

## How it works

1. **Start a session** — the app captures audio from your default microphone.
2. **Real-time detection** — audio chunks are classified using the YAMNet TFLite model running locally. When a cough is detected with sufficient confidence, the counter increments.
3. **Session history** — completed sessions (timestamp, duration, cough count, rate) are saved to a local SQLite database and displayed in a table.
4. **Session detail** — double-click any past session to see a chart of coughs over time.

## Setup

```bash
# Clone the repo
git clone https://github.com/christiantrummer/cough-counter.git
cd cough-counter

# Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On first run, the YAMNet model (~4 MB) is downloaded automatically.

## Usage

```bash
python3 app.py
```

Click **Start Session** to begin detecting coughs. Click **Stop Session** to end and save the session.

## Requirements

- Python 3.10+
- macOS (microphone access required — grant permission when prompted)
- Dependencies: `mediapipe`, `sounddevice`, `numpy`, `matplotlib`

## Configuration

Detection parameters can be adjusted in `audio_classifier.py`:

| Parameter | Default | Description |
|---|---|---|
| `COUGH_SCORE_THRESHOLD` | `0.35` | Minimum confidence score to count a cough |
| `DEBOUNCE_SECONDS` | `1.5` | Cooldown between detections to avoid double-counting |
