import os
import json
import base64
import asyncio
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Configuration & Logging
# ---------------------------------------------------------------------------
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ungal_vijay_gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
if SUPABASE_URL.endswith("/rest/v1"):
    SUPABASE_URL = SUPABASE_URL[:-8].rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
GEMINI_LIVE_VOICE = os.getenv("GEMINI_LIVE_VOICE", "Kore")

PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Load System Prompt
prompt_candidates = [
    os.path.join(os.path.dirname(__file__), "system_prompt.txt"),
    os.path.join(os.getcwd(), "system_prompt.txt"),
    os.path.join(os.path.dirname(__file__), "..", "system_prompt.txt"),
]
SYSTEM_PROMPT = ""
for p in prompt_candidates:
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                SYSTEM_PROMPT = f.read()
            logger.info(f"Loaded system prompt successfully from: {p}")
            break
        except Exception as e:
            logger.warning(f"Could not load prompt from {p}: {e}")

if not SYSTEM_PROMPT:
    logger.warning("system_prompt.txt not found. Using default prompt.")
    SYSTEM_PROMPT = "You are the Tamil Nadu Government Welfare Schemes Voice AI helpline counselor. Speak natural spoken Tanglish."

# ---------------------------------------------------------------------------
# 2. Database Repository (Supabase with In-Memory Startup Cache)
# ---------------------------------------------------------------------------
class SchemeRepository:
    """
    Preloads all 25 schemes into server memory on boot to guarantee
    sub-millisecond lookups and zero network lag during phone calls.
    """
    def __init__(self, url: str, key: str):
        self.client: Optional[Client] = None
        if url and key:
            try:
                self.client = create_client(url, key)
                logger.info("Supabase client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
        else:
            logger.warning("Supabase credentials missing. Running in offline mock mode.")

        self.cached_schemes: List[Dict[str, Any]] = []

    async def warm_up_cache(self):
        """Loads all schemes from Supabase into server memory at boot time."""
        if not self.client:
            return
        try:
            response = await asyncio.to_thread(
                lambda: self.client.table("schemes").select("*").execute()
            )
            self.cached_schemes = response.data or []
            logger.info(f"Database cache initialized: {len(self.cached_schemes)} schemes loaded into memory.")
        except Exception as e:
            logger.error(f"Error fetching schemes from Supabase during warmup: {e}")

    def search_schemes(
        self,
        query: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        marital_status: Optional[str] = None,
        income: Optional[int] = None,
        community: Optional[str] = None,
        special_condition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filters schemes in server memory matching the Big 4 triage fields."""
        results = self.cached_schemes

        # 1. Keyword search across names, benefits, and tags
        if query and query.strip():
            terms = query.lower().strip().split()
            filtered = []
            for s in results:
                searchable_text = " ".join([
                    s.get("name_en") or "",
                    s.get("name_ta") or "",
                    s.get("brief_benefit_en") or "",
                    s.get("objective_en") or "",
                    s.get("benefits_en") or "",
                    " ".join(s.get("special_conditions") or [])
                ]).lower()
                if any(t in searchable_text for t in terms):
                    filtered.append(s)
            results = filtered

        # 2. Gender filtering
        if gender:
            g_val = gender.lower()
            results = [
                s for s in results
                if not s.get("gender") or "any" in s["gender"] or g_val in s["gender"]
            ]

        # 3. Age filtering (min_age <= age <= max_age)
        if age is not None:
            def check_age(s):
                min_a = s.get("min_age")
                max_a = s.get("max_age")
                if min_a is not None and age < min_a:
                    return False
                if max_a is not None and age > max_a:
                    return False
                return True
            results = [s for s in results if check_age(s)]

        # 4. Income filtering (scheme max_income >= caller income)
        if income is not None:
            def check_income(s):
                max_inc = s.get("max_income")
                if max_inc is not None and income > max_inc:
                    return False
                return True
            results = [s for s in results if check_income(s)]

        # 5. Marital Status filtering
        if marital_status:
            m_val = marital_status.lower()
            results = [
                s for s in results
                if not s.get("marital_status") or "any" in s["marital_status"] or m_val in s["marital_status"]
            ]

        # 6. Community filtering (sc, st, scc, bc, mbc, fc)
        if community:
            c_val = community.lower()
            results = [
                s for s in results
                if not s.get("community") or "any" in s["community"] or c_val in s["community"]
            ]

        # 7. Special condition tags
        if special_condition:
            cond_val = special_condition.lower()
            results = [
                s for s in results
                if s.get("special_conditions") and cond_val in [c.lower() for c in s["special_conditions"]]
            ]

        return results

    def get_scheme_by_id_or_name(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Returns the full record for a specific scheme by ID or name."""
        ident_lower = identifier.lower().strip()
        for s in self.cached_schemes:
            if s.get("id") == identifier:
                return s
            if ident_lower in (s.get("name_en") or "").lower():
                return s
            if ident_lower in (s.get("name_ta") or "").lower():
                return s
        return None

db_repo = SchemeRepository(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# 3. Tool Declarations for Gemini Live API
# ---------------------------------------------------------------------------
SEARCH_SCHEMES_DECLARATION = {
    "name": "search_schemes",
    "description": (
        "Searches Tamil Nadu welfare schemes using caller keywords or the Big 4 triage fields "
        "(age, gender, marital status, income, community). Call this when the caller mentions a specific need, "
        "a scheme name, or answers triage questions."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "Scheme name keyword or benefit keyword, e.g. 'marriage', 'pudhumai penn', 'goat', 'sewing machine', 'loan'."
            },
            "age": {
                "type": "INTEGER",
                "description": "Age of the beneficiary."
            },
            "gender": {
                "type": "STRING",
                "description": "Gender of the beneficiary: 'female', 'male', 'transgender'."
            },
            "marital_status": {
                "type": "STRING",
                "description": "Marital/family status of beneficiary: 'single', 'married', 'widow', 'deserted', 'destitute'."
            },
            "income": {
                "type": "INTEGER",
                "description": "Annual family income in Rupees, e.g. 72000 or 250000."
            },
            "community": {
                "type": "STRING",
                "description": "Caste category if relevant: 'sc', 'st', 'scc', 'bc', 'mbc', 'fc'."
            },
            "special_condition": {
                "type": "STRING",
                "description": "Special attribute such as 'orphan', 'inter_caste', 'tailoring_trained', 'landless_agricultural_labourer'."
            }
        }
    }
}

GET_SCHEME_DETAILS_DECLARATION = {
    "name": "get_scheme_details",
    "description": (
        "Retrieves full application procedure, required documents, and contact details "
        "for a specific scheme ID or name once the caller chooses to explore it."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "scheme_identifier": {
                "type": "STRING",
                "description": "Scheme UUID or scheme English/Tamil name."
            }
        },
        "required": ["scheme_identifier"]
    }
}

# ---------------------------------------------------------------------------
# 4. Tool Execution Handler (Dual-Mode Single-Match Preloader)
# ---------------------------------------------------------------------------
async def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes function calls requested by Gemini.
    - Mode 1: If 1 match, bundle 1-liner + preload full documents so follow-up is instant.
    - Mode 2: If 2-4 matches, return lightweight summaries so the caller can choose first.
    """
    if tool_name == "search_schemes":
        matches = db_repo.search_schemes(
            query=args.get("query"),
            age=args.get("age"),
            gender=args.get("gender"),
            marital_status=args.get("marital_status"),
            income=args.get("income"),
            community=args.get("community"),
            special_condition=args.get("special_condition")
        )

        total_found = len(matches)

        if total_found == 0:
            return {
                "count": 0,
                "message": "No schemes matched the provided criteria."
            }

        # MODE 1: Exactly 1 Match -> Bundle 1-line benefit AND preload full details!
        if total_found == 1:
            scheme = matches[0]
            return {
                "count": 1,
                "mode": "single_match_preloaded",
                "scheme": {
                    "id": scheme.get("id"),
                    "name_en": scheme.get("name_en"),
                    "name_ta": scheme.get("name_ta"),
                    "brief_benefit_en": scheme.get("brief_benefit_en"),
                    "preloaded_details": {
                        "benefits": scheme.get("benefits_en"),
                        "application_procedure": scheme.get("application_procedure_en"),
                        "required_documents_ta": scheme.get("required_documents_ta"),
                        "required_documents_en": scheme.get("required_documents_en"),
                        "portal_url": scheme.get("portal_url"),
                        "helpline": scheme.get("helpline"),
                        "officer_contact": scheme.get("officer_contact_ta") or scheme.get("officer_contact_en")
                    }
                }
            }

        # MODE 2: Multiple Matches (2 to 4) -> Lightweight summary options only
        summary_list = []
        for s in matches[:4]:
            summary_list.append({
                "id": s.get("id"),
                "name_en": s.get("name_en"),
                "name_ta": s.get("name_ta"),
                "brief_benefit_en": s.get("brief_benefit_en")
            })

        return {
            "count": total_found,
            "mode": "multiple_matches",
            "matches": summary_list,
            "instruction": "Present the brief benefits of these options and ask the caller which one to explore."
        }

    elif tool_name == "get_scheme_details":
        ident = args.get("scheme_identifier", "")
        scheme = db_repo.get_scheme_by_id_or_name(ident)
        if not scheme:
            return {"error": f"Scheme '{ident}' not found in database."}

        return {
            "id": scheme.get("id"),
            "name_en": scheme.get("name_en"),
            "name_ta": scheme.get("name_ta"),
            "benefits": scheme.get("benefits_en"),
            "application_procedure": scheme.get("application_procedure_en"),
            "required_documents_ta": scheme.get("required_documents_ta"),
            "required_documents_en": scheme.get("required_documents_en"),
            "portal_url": scheme.get("portal_url"),
            "helpline": scheme.get("helpline"),
            "officer_contact": scheme.get("officer_contact_ta") or scheme.get("officer_contact_en")
        }

    return {"error": f"Unknown tool: {tool_name}"}

# ---------------------------------------------------------------------------
# 5. FastAPI Application & Lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Ungal Vijay Backend Server (Gemini 3.8 Live Engine)...")
    await db_repo.warm_up_cache()
    yield
    logger.info("Ungal Vijay Backend Server shutdown complete.")

app = FastAPI(title="Ungal Vijay Voice AI Backend (Gemini 3.8 Live)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_candidates = [
    os.environ.get("FRONTEND_DIR"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend")),
    os.path.abspath(os.path.join(os.getcwd(), "frontend")),
]
frontend_dir = None
for candidate in frontend_candidates:
    if candidate and os.path.exists(candidate) and os.path.isdir(candidate):
        frontend_dir = candidate
        break

if frontend_dir:
    logger.info(f"Serving frontend from: {frontend_dir}")
    img_dir = os.path.join(frontend_dir, "img")
    if os.path.exists(img_dir):
        app.mount("/img", StaticFiles(directory=img_dir), name="img")
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def serve_index():
    if frontend_dir:
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return {"message": "Ungal Vijay Voice AI is running. Frontend index.html not found."}

@app.get("/index.html")
async def serve_index_html():
    return await serve_index()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "gemini-3.8-live",
        "schemes_cached": len(db_repo.cached_schemes),
        "supabase_connected": db_repo.client is not None,
        "gemini_api_key_configured": bool(GEMINI_API_KEY)
    }

@app.get("/debug-gemini")
async def debug_gemini():
    import traceback
    import websockets
    info = {
        "gemini_api_key_len": len(GEMINI_API_KEY),
        "gemini_api_key_prefix": GEMINI_API_KEY[:6] if GEMINI_API_KEY else None,
        "gemini_live_model": GEMINI_LIVE_MODEL,
        "google_genai_version": getattr(genai, "__version__", "unknown")
    }

    # Probe 1: SDK connect with v1beta (default)
    try:
        client_v1beta = genai.Client(api_key=GEMINI_API_KEY)
        cfg = types.LiveConnectConfig(response_modalities=["AUDIO"])
        async with client_v1beta.aio.live.connect(model="gemini-3.8-live", config=cfg) as sess:
            info["probe_sdk_v1beta"] = "SUCCESS"
    except Exception as e:
        info["probe_sdk_v1beta"] = f"{type(e).__name__}: {str(e)}"

    # Probe 2: SDK connect with v1alpha
    try:
        client_v1alpha = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(api_version="v1alpha"))
        cfg = types.LiveConnectConfig(response_modalities=["AUDIO"])
        async with client_v1alpha.aio.live.connect(model="gemini-3.8-live", config=cfg) as sess:
            info["probe_sdk_v1alpha"] = "SUCCESS"
    except Exception as e:
        info["probe_sdk_v1alpha"] = f"{type(e).__name__}: {str(e)}"

    # Probe 3: Raw WebSocket with ?key= on v1beta
    uri_v1beta = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GEMINI_API_KEY}"
    try:
        async with websockets.connect(uri_v1beta) as ws:
            setup = {"setup": {"model": "models/gemini-3.8-live", "generationConfig": {"responseModalities": ["AUDIO"]}}}
            await ws.send(json.dumps(setup))
            resp = await asyncio.wait_for(ws.recv(), timeout=4.0)
            info["probe_raw_ws_v1beta"] = f"SUCCESS: {resp[:80]}"
    except Exception as e:
        info["probe_raw_ws_v1beta"] = f"{type(e).__name__}: {str(e)}"

    # Probe 4: Raw WebSocket with ?key= on v1alpha
    uri_v1alpha = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={GEMINI_API_KEY}"
    try:
        async with websockets.connect(uri_v1alpha) as ws:
            setup = {"setup": {"model": "models/gemini-3.8-live", "generationConfig": {"responseModalities": ["AUDIO"]}}}
            await ws.send(json.dumps(setup))
            resp = await asyncio.wait_for(ws.recv(), timeout=4.0)
            info["probe_raw_ws_v1alpha"] = f"SUCCESS: {resp[:80]}"
    except Exception as e:
        info["probe_raw_ws_v1alpha"] = f"{type(e).__name__}: {str(e)}"

    return info

# ---------------------------------------------------------------------------
# 6. WebSocket Telephony Bridge (Browser <-> Python <-> Gemini Live)
# ---------------------------------------------------------------------------
@app.websocket("/ws/call")
async def websocket_call_endpoint(client_ws: WebSocket):
    """
    Bidirectional streaming bridge:
    - Receives microphone PCM audio (16kHz) from browser and forwards to Gemini Live.
    - Handles tool calling locally against the cached 25 schemes.
    - Relays synthesized 24kHz voice bytes and live subtitles back to browser.
    """
    await client_ws.accept()
    logger.info("Caller connected from browser client.")

    if not GEMINI_API_KEY:
        await client_ws.send_json({
            "type": "error",
            "message": "GEMINI_API_KEY is missing on the server. Please check your .env file."
        })
        await client_ws.close()
        return

    try:
        # Initialize Google GenAI client
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)

        # Configure session
        live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=GEMINI_LIVE_VOICE
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=SYSTEM_PROMPT)]
        ),
        tools=[{
            "function_declarations": [
                SEARCH_SCHEMES_DECLARATION,
                GET_SCHEME_DETAILS_DECLARATION
            ]
        }]
    )

        # Robust model resolution: try configured model, then standard live models
        candidates = [
            GEMINI_LIVE_MODEL,
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-realtime-exp",
            "gemini-2.5-flash"
        ]
        candidates = list(dict.fromkeys(candidates))

        connected_cm = None
        session = None
        active_model = None

        for cand in candidates:
            try:
                cm = gemini_client.aio.live.connect(model=cand, config=live_config)
                session = await cm.__aenter__()
                connected_cm = cm
                active_model = cand
                logger.info(f"Connected to Gemini Live session using model '{cand}' with voice '{GEMINI_LIVE_VOICE}'.")
                break
            except Exception as e:
                logger.warning(f"Model candidate '{cand}' rejected ({e}). Trying next...")

        if not connected_cm or not session:
            raise RuntimeError(f"Gemini Live rejected connection on all candidate models ({candidates}). Check API key access in Google AI Studio.")

        try:
            # Trigger opening greeting immediately upon connection
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="Start the call now by speaking the official opening greeting exactly as instructed."
                    )]
                ),
                turn_complete=True
            )

            # Stream A: Forward client microphone audio / commands to Gemini
            async def forward_client_to_gemini():
                try:
                    while True:
                        message = await client_ws.receive()
                        if "bytes" in message and message["bytes"]:
                            # Raw PCM 16kHz binary chunk from browser
                            pcm_chunk = message["bytes"]
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=pcm_chunk,
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )
                        elif "text" in message and message["text"]:
                            event = json.loads(message["text"])
                            event_type = event.get("type")

                            if event_type == "audio_chunk":
                                raw_base64 = event.get("audio", "")
                                if raw_base64:
                                    pcm_bytes = base64.b64decode(raw_base64)
                                    await session.send_realtime_input(
                                        audio=types.Blob(
                                            data=pcm_bytes,
                                            mime_type="audio/pcm;rate=16000"
                                        )
                                    )
                            elif event_type == "call.hangup":
                                logger.info("Caller hung up from browser.")
                                break
                except WebSocketDisconnect:
                    logger.info("Browser WebSocket disconnected.")
                except Exception as e:
                    logger.error(f"Error in client-to-gemini stream: {e}")

            # Stream B: Receive Gemini audio/subtitles + execute tools
            async def forward_gemini_to_client():
                try:
                    async for response in session.receive():
                        # 1. Handle Tool / Function Calls
                        if response.tool_call:
                            function_responses = []
                            for fc in response.tool_call.function_calls:
                                logger.info(f"Gemini invoked tool: {fc.name} with args {fc.args}")
                                
                                await client_ws.send_json({
                                    "type": "tool_call_started",
                                    "tool": fc.name,
                                    "arguments": fc.args
                                })

                                # Execute locally in 0.2ms
                                result = await execute_tool(fc.name, fc.args)

                                await client_ws.send_json({
                                    "type": "tool_call_completed",
                                    "tool": fc.name,
                                    "result": result
                                })

                                function_responses.append(
                                    types.FunctionResponse(
                                        name=fc.name,
                                        id=fc.id,
                                        response={"result": result}
                                    )
                                )

                            # Return tool output back to Gemini
                            await session.send_tool_response(function_responses=function_responses)

                        # 2. Handle Server Audio & Subtitles
                        if response.server_content:
                            # Interruption event: User started talking, Gemini cut audio
                            if response.server_content.interrupted:
                                await client_ws.send_json({"type": "interrupted"})

                            model_turn = response.server_content.model_turn
                            if model_turn:
                                for part in model_turn.parts:
                                    # Audio chunk (24kHz PCM)
                                    if part.inline_data and part.inline_data.data:
                                        audio_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                                        await client_ws.send_json({
                                            "type": "audio_delta",
                                            "delta": audio_b64
                                        })
                                    # Text transcription / subtitle
                                    elif part.text:
                                        await client_ws.send_json({
                                            "type": "ai_transcript_delta",
                                            "delta": part.text
                                        })

                except Exception as e:
                    logger.error(f"Error in gemini-to-client stream: {e}")

            # Run both streaming tasks concurrently
            await asyncio.gather(
                forward_client_to_gemini(),
                forward_gemini_to_client(),
                return_exceptions=True
            )
        finally:
            if connected_cm:
                await connected_cm.__aexit__(None, None, None)

    except Exception as e:
        logger.error(f"Telephony bridge failed: {e}")
        await client_ws.send_json({
            "type": "error",
            "message": f"Connection to Gemini voice service failed: {str(e)}"
        })
    finally:
        try:
            await client_ws.close()
        except Exception:
            pass
        logger.info("Call session ended.")

# ---------------------------------------------------------------------------
# 7. Runner Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
