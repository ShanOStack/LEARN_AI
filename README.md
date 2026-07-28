# 🔍 AI Search Agent

An AI-powered search assistant built with **Streamlit**, **LangChain**, and **Groq LLMs**. It searches the web (DuckDuckGo, Wikipedia, Arxiv), answers in detailed paragraphs + key-point bullets, and generates related images, YouTube videos, a hierarchical mind map, and AI-written short notes — all with voice input/output and TXT/PDF export.

---

## ✨ Features

- 🔎 **Multi-source search** — DuckDuckGo, Wikipedia, and Arxiv, orchestrated by a LangChain agent
- 🧠 **Groq LLM** — fast responses using `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`
- 📝 **Structured answers** — 3–4 detailed paragraphs + a "Key Points" bullet summary
- 🎙️ **Voice input** — ask questions by speaking (mic button in the chat bar)
- 🔊 **Voice output** — listen to the answer (text-to-speech)
- 🖼️ **Related images** — Google Custom Search (optional), DuckDuckGo, Openverse, and Wikipedia thumbnails as fallbacks
- 📺 **YouTube results** — related videos with thumbnails, fetched via `yt-dlp` (no API key needed)
- 🧭 **Mind map** — auto-generated, multi-shape hierarchical diagram of the answer's key ideas
- 📋 **AI short notes** — a colorful, sectioned notes card (5 headings × 5 points) with its own topic diagram, downloadable as PNG
- 📥 **Export** — download any answer as TXT or PDF
- 🌙 **Dark themed UI** with card-based layout

---

## 🛠️ Tech Stack

| Purpose | Library |
|---|---|
| UI | Streamlit |
| LLM orchestration | LangChain + `langchain-groq` |
| LLM provider | Groq |
| Web/Wiki/Arxiv search | `ddgs`, `wikipedia`, `arxiv` |
| YouTube search | `yt-dlp` |
| Voice input | `streamlit-mic-recorder`, `SpeechRecognition`, `PyAudio` |
| Voice output | `gTTS` |
| Mind map / diagrams | `networkx`, `matplotlib` |
| PDF export | `reportlab` |
| Image processing | `Pillow` |

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS/Linux
source myenv/bin/activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install system dependencies

**PyAudio** (voice input) usually installs directly via pip. If it fails on Windows with a build error, try:
```bash
pip install pipwin
pipwin install pyaudio
```

**Graphviz** (used by the diagram libraries) needs the system tool installed separately:
- Windows: download the installer from [graphviz.org/download](https://graphviz.org/download/), install it, and check **"Add Graphviz to the system PATH for all users"**
- macOS: `brew install graphviz`
- Linux: `sudo apt install graphviz portaudio19-dev python3-pyaudio`

Verify with:
```bash
dot -V
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional — enables real Google Images results (100 free searches/day)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
```

- Get a free Groq API key: [console.groq.com](https://console.groq.com)
- Get a Google API key + Custom Search Engine ID (optional):
  [console.cloud.google.com](https://console.cloud.google.com) (enable **Custom Search API**) +
  [programmablesearchengine.google.com](https://programmablesearchengine.google.com) (turn on **Image search** and **Search the entire web**)

If the Google keys aren't set, the app automatically falls back to DuckDuckGo → Openverse → Wikipedia for images.

---

## ▶️ Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not committed — see .gitignore)
└── .streamlit/
    └── config.toml          # Dark theme configuration
```

---

## ⚠️ Troubleshooting

- **`JSONDecodeError` from Wikipedia tool** — Wikipedia's API occasionally rejects requests without a proper User-Agent, or rate-limits. The app already retries and falls back gracefully; if it persists, it usually resolves on its own after a short wait.
- **Rate limit / 429 error from Groq** — Groq's free tier has a daily token cap per model. Switch to `llama-3.1-8b-instant` in the sidebar (lighter model) or wait for the daily quota to reset.
- **`Parsing error: Invalid Format`** — occasionally the LLM doesn't follow the exact agent format. The app automatically retries with corrective instructions; if it keeps happening, try the `llama-3.1-8b-instant` model, which tends to follow the format more consistently.
- **PyAudio install fails on Windows** — use `pipwin install pyaudio` (see Installation above).
- **`dot: command not found`** — Graphviz's system binary isn't on PATH; reinstall and check the PATH option, or add `C:\Program Files\Graphviz\bin` to your PATH manually.

---

## 📄 License

This project is open source — feel free to modify and use it for your own purposes.

---

## 🙋 Author

Developed by **Shanteshwar Ojha**.
