# WhatsApp Group Insights

Local app that analyzes a WhatsApp group chat export and summarizes stock-related discussion. Optional local LLM support via Ollama.

## What you get
- Summary of the group discussion
- Stocks mentioned and how often
- Buy/sell/watch signals (heuristic)
- Optional LLM-based summary and reasons (Ollama)

## How to export a chat (WhatsApp Desktop on macOS)
1. Open the group chat.
2. Open the group info panel.
3. Choose **Export Chat**.
4. Save as a `.txt` file (without media), or choose **Include Media** to get a `.zip`.

## Run locally
```bash
cd "/Users/tanuj/Documents/New project/whatsapp-group-insights"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Media support (local + free)
Supported media types:
- Images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.tiff`, `.bmp` (OCR via Tesseract)
- PDFs: `.pdf` (text extraction)
- Docs: `.docx`
- Audio/Video: `.mp3`, `.m4a`, `.wav`, `.aac`, `.mp4`, `.mov`, `.mkv` (local transcription via Whisper)

Install system dependencies:
```bash
brew install tesseract ffmpeg
```

Optional: install Whisper for audio/video transcription:
```bash
pip install openai-whisper
```

## Optional: use a local LLM (Ollama)
1. Install Ollama: https://ollama.com
2. Pull a model:
```bash
ollama pull llama3.1
```
3. In the app, keep **Use Ollama** checked and set the model name to `llama3.1`.

## Notes
- This app reads exported files only. It does **not** read WhatsApp directly.
- For media OCR/transcription, upload the `.zip` export.
- Ollama is optional. Without it, you still get heuristic stock mentions.
