# உங்கள் விஜய் (Ungal Vijay) v2: Solved Gemini Live Architecture

We have completely built **v2** from scratch in `D:\ungal vijay\v2\` using Google's official, battle-tested reference architecture (`gemini-live-ephemeral-tokens-websocket`).

---

## 1. What Changed from v1 to v2?

| Component | v1 (Brittle Proxy Architecture) | v2 (Google's Solved Direct WebSocket Architecture) |
| :--- | :--- | :--- |
| **Connection Path** | Browser $\rightarrow$ Railway Python Server $\rightarrow$ Gemini | **Browser $\rightarrow$ Google Gemini Live API directly** via WebSocket (`wss://generativelanguage.googleapis.com/...`) |
| **Authentication** | Server kept API key and proxied everything | Server issues short-lived **Ephemeral Token** (`POST /api/token`) |
| **Audio Pipeline** | ScriptProcessor resampling in Python generator | Google's **`AudioCaptureProcessor`** (16kHz PCM worklet) & **`PCMProcessor`** (24kHz zero-allocation playback worklet) |
| **Gemini Going Silent** | Python `session.receive()` terminated after first turn | Handled natively by browser WebSocket; session never terminates prematurely |
| **Equalizer Movement** | Artificial sine wave formula moved even during silence | **100% still/flat on silence**; dynamically pulses only to real vocal energy |
| **Subtitles** | Missing due to unconfigured transcription | Streams in real-time from `outputAudioTranscription` |
| **Volume Control** | Fixed direct destination (rocker didn't work) | Software **GainNode** connected to volume slider and mute toggle dock |
| **Call Teardown** | Audio frames leaked after hangup | Instant microphone hardware shutdown (`track.stop()`) & WebSocket close |
| **Welfare Schemes** | Python proxy bottleneck | Client executes function calls (`search_schemes`, `get_scheme_details`) against cached backend and sends `toolResponse` |

---

## 2. Directory Structure of v2

```
D:\ungal vijay\v2\
├── backend/
│   ├── app.py                # FastAPI server (Ephemeral tokens, Scheme cache, Tool lookup endpoints)
│   ├── system_prompt.txt     # Official v3.0 Tanglish Counselor prompt
│   ├── requirements.txt      # Dependencies: fastapi, uvicorn, google-genai, supabase, python-dotenv
│   ├── Procfile              # Railway start command (web: uvicorn app:app --host 0.0.0.0 --port $PORT)
│   └── .env.example          # Environment variables template
└── frontend/
    ├── index.html            # Exact UI styling, 72px avatar, gold shimmer, velvet buttons, volume dock
    ├── img/
    │   └── vijay.jpg         # High-res official portrait
    └── audio-processors/
        ├── capture.worklet.js  # 16kHz PCM microphone capture worklet
        └── playback.worklet.js # 24kHz zero-allocation PCM playback worklet
```

---

## 3. How to Run Locally

To test **v2** locally on your machine:

1. Open PowerShell and navigate to the backend directory:
   ```powershell
   cd "D:\ungal vijay\v2\backend"
   ```
2. Create a `.env` file with your credentials:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   PORT=8000
   ```
3. Start the server:
   ```powershell
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
4. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```
5. Click **"இணைப்பைத் தொடங்கவும்"** (Start Call) to test voice conversation with Vijay!

---

## 4. Verification Results

All automated tests executed against `D:\ungal vijay\v2\` passed:
- `GET /health` $\rightarrow$ `200 OK`
- `GET /api/config` $\rightarrow$ `200 OK` (Loaded model, voice, system prompt, and tools)
- `POST /api/search_schemes` $\rightarrow$ `200 OK` (Tested keyword scoring and exact token filtering)
- `POST /api/get_scheme_details` $\rightarrow$ `200 OK` (Loaded scheme details)
- `GET /` $\rightarrow$ `200 OK` (Served `index.html`)
- `GET /audio-processors/capture.worklet.js` $\rightarrow$ `200 OK`
- `GET /audio-processors/playback.worklet.js` $\rightarrow$ `200 OK`
- `GET /img/vijay.jpg` $\rightarrow$ `200 OK`
