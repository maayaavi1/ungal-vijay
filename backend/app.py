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

        # 1. Intelligent keyword search with stop-word filtering and relevance scoring
        if query and isinstance(query, str) and query.strip():
            raw_terms = query.lower().strip().split()
            STOP_WORDS = {
                "scheme", "schemes", "thittam", "thittangal", "for", "the", "a", "an",
                "in", "of", "and", "to", "arasu", "government", "tamil", "nadu", "service",
                "pathi", "solla", "sollunga", "irukka", "kedaikkuma", "details", "enna",
                "sevai", "sevai-kku", "help", "helpline", "scheme-kku", "scheme-gal", "about"
            }
            meaningful_terms = [t for t in raw_terms if t not in STOP_WORDS and len(t) > 2]

            # Only perform keyword scoring if caller used specific meaningful terms.
            # If query had ONLY generic stop words (e.g. 'government scheme', 'arasu thittam'),
            # do NOT fall back to stop words to prevent false-positive scoring across all 25 schemes.
            if meaningful_terms:
                scored = []
                for s in results:
                    score = 0
                    name_text = f"{s.get('name_en') or ''} {s.get('name_ta') or ''}".lower()
                    benefit_text = f"{s.get('brief_benefit_en') or ''} {' '.join(s.get('special_conditions') or [])}".lower()
                    body_text = f"{s.get('objective_en') or ''} {s.get('benefits_en') or ''}".lower()

                    for t in meaningful_terms:
                        if t in name_text:
                            score += 5  # Direct match on official scheme name
                        elif t in benefit_text:
                            score += 3  # Direct match on 1-line benefit or tag
                        elif t in body_text:
                            score += 1  # Match in detailed description

                    if score > 0:
                        scored.append((score, s))

                # Rank the highest-scoring schemes at the top
                scored.sort(key=lambda x: x[0], reverse=True)
                results = [s for score, s in scored]

        # Helper for exact category token matching (prevents 'male' in 'female' or 'bc' in 'mbc')
        def match_tokens(field_val, target_val):
            if not field_val or not target_val:
                return True
            if isinstance(field_val, list):
                categories = set(str(x).strip().lower() for x in field_val if str(x).strip())
            else:
                categories = set(x.strip().lower() for x in str(field_val).replace('/', ',').split(',') if x.strip())
            if 'any' in categories or 'all' in categories:
                return True
            if isinstance(target_val, list):
                targets = set(str(x).strip().lower() for x in target_val if str(x).strip())
            else:
                targets = set(x.strip().lower() for x in str(target_val).replace('/', ',').split(',') if x.strip())
            # Exact token intersection prevents substring collisions ('male' != 'female', 'bc' != 'mbc')
            return bool(targets & categories)

        # 2. Gender filtering (exact category match)
        if gender:
            results = [s for s in results if match_tokens(s.get("gender"), gender)]

        # 3. Age filtering (min_age <= age <= max_age) with safe type casting
        if age is not None:
            try:
                age_val = int(age)
            except (ValueError, TypeError):
                age_val = None

            if age_val is not None:
                def check_age(s):
                    min_a = s.get("min_age")
                    max_a = s.get("max_age")
                    try:
                        if min_a is not None and age_val < int(min_a):
                            return False
                    except (ValueError, TypeError):
                        pass
                    try:
                        if max_a is not None and age_val > int(max_a):
                            return False
                    except (ValueError, TypeError):
                        pass
                    return True
                results = [s for s in results if check_age(s)]

        # 4. Income filtering (scheme max_income >= caller income) with safe type casting
        if income is not None:
            try:
                income_val = int(income)
            except (ValueError, TypeError):
                income_val = None

            if income_val is not None:
                def check_income(s):
                    max_inc = s.get("max_income")
                    try:
                        if max_inc is not None and income_val > int(max_inc):
                            return False
                    except (ValueError, TypeError):
                        pass
                    return True
                results = [s for s in results if check_income(s)]

        # 5. Marital Status filtering (exact category match)
        if marital_status:
            results = [s for s in results if match_tokens(s.get("marital_status"), marital_status)]

        # 6. Community filtering (exact category match)
        if community:
            results = [s for s in results if match_tokens(s.get("community"), community)]

        # 7. Special condition tags
        if special_condition:
            cond_val = special_condition.lower()
            results = [
                s for s in results
                if s.get("special_conditions") and cond_val in [c.lower() for c in s["special_conditions"]]
            ]

        return results

    def get_scheme_by_id_or_name(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Returns the full record for a specific scheme by ID or name (exact match first)."""
        if not identifier:
            return None
        ident_lower = identifier.lower().strip()
        # Pass 1: Exact matches
        for s in self.cached_schemes:
            if s.get("id") == identifier:
                return s
            if (s.get("name_en") or "").lower().strip() == ident_lower:
                return s
            if (s.get("name_ta") or "").lower().strip() == ident_lower:
                return s
        # Pass 2: Substring matches
        for s in self.cached_schemes:
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
    "behavior": "BLOCKING",
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
    "behavior": "BLOCKING",
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
                        "portal_url": (scheme.get("portal_url") or "").replace("https://", "").replace("http://", "").rstrip("/"),
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
            "count": len(summary_list),
            "total_matches_in_database": total_found,
            "mode": "multiple_matches",
            "matches": summary_list,
            "instruction": "Present the brief benefits of these top matching options and ask the caller which one they want to explore."
        }

    elif tool_name == "get_scheme_details":
        ident = str(args.get("scheme_identifier") or args.get("scheme_name") or args.get("name") or args.get("query") or "").strip()
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
            "portal_url": (scheme.get("portal_url") or "").replace("https://", "").replace("http://", "").rstrip("/"),
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
        # Initialize Google GenAI client with v1alpha (verified successful for live audio with AQ Auth keys)
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(api_version="v1alpha")
        )

        # Configure session with audio transcription for real-time subtitles
        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
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
            # Stream A: Forward client microphone audio / commands to Gemini Live
            async def forward_client_to_gemini():
                audio_packets_sent = 0
                try:
                    while True:
                        message = await client_ws.receive()
                        msg_type = message.get("type")
                        if msg_type == "websocket.disconnect":
                            logger.info("Browser WebSocket disconnected.")
                            break

                        pcm_bytes = None
                        if message.get("bytes"):
                            pcm_bytes = message["bytes"]
                        elif message.get("text"):
                            try:
                                event = json.loads(message["text"])
                            except Exception:
                                continue
                            event_type = event.get("type")
                            if event_type == "audio_chunk":
                                raw_base64 = event.get("audio", "")
                                if raw_base64:
                                    pcm_bytes = base64.b64decode(raw_base64)
                            elif event_type == "call.hangup":
                                logger.info("Caller hung up from browser.")
                                break

                        if pcm_bytes:
                            # Forward exact protobuf realtimeInput structure to Gemini Live
                            b64_audio = base64.b64encode(pcm_bytes).decode("utf-8")
                            realtime_payload = {
                                "realtimeInput": {
                                    "mediaChunks": [
                                        {
                                            "mimeType": "audio/pcm;rate=16000",
                                            "data": b64_audio
                                        }
                                    ]
                                }
                            }
                            # Send directly to the active websocket connection
                            if hasattr(session, '_ws') and session._ws:
                                await session._ws.send(json.dumps(realtime_payload))
                            else:
                                await session.send_realtime_input(
                                    media=types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000")
                                )
                            audio_packets_sent += 1
                            if audio_packets_sent % 50 == 0:
                                logger.info(f"Streamed {audio_packets_sent} mic packets to Gemini Live.")
                except (WebSocketDisconnect, asyncio.CancelledError):
                    logger.info("Browser WebSocket disconnected.")
                except Exception as e:
                    logger.error(f"Error in client-to-gemini stream: {e}")

            # Stream B: Receive Gemini audio/subtitles + execute tools (continuous multi-turn stream)
            async def forward_gemini_to_client():
                try:
                    while True:
                        # Direct call to session._receive() persists indefinitely across all conversational turns
                        response = await session._receive()
                        if not response:
                            continue

                        # 1. Handle Tool / Function Calls
                        if response.tool_call:
                            function_responses = []
                            for fc in response.tool_call.function_calls:
                                args = fc.args or {}
                                logger.info(f"Gemini invoked tool: {fc.name} with args {args}")
                                try:
                                    result = await execute_tool(fc.name, args)
                                except Exception as e:
                                    logger.exception(f"Tool {fc.name} execution failed: {e}")
                                    result = {"error": f"Data lookup error: {str(e)}"}

                                function_responses.append(
                                    types.FunctionResponse(
                                        name=fc.name,
                                        id=fc.id,
                                        response={"result": result}
                                    )
                                )

                            # Return tool output back to Gemini
                            await session.send_tool_response(function_responses=function_responses)

                        sc = response.server_content
                        if not sc:
                            continue

                        # Interruption event: User started talking, Gemini cut audio
                        if sc.interrupted:
                            logger.info("Gemini playback interrupted by caller speech.")
                            await client_ws.send_json({"type": "interrupted"})

                        # Log caller speech when transcribed (verifies mic pipeline)
                        if sc.input_transcription and sc.input_transcription.text:
                            logger.info(f"Caller said: {sc.input_transcription.text}")

                        # Extract real-time speech subtitles from output_transcription
                        if sc.output_transcription and sc.output_transcription.text:
                            logger.info(f"AI Subtitle (output_transcription): {sc.output_transcription.text}")
                            await client_ws.send_json({
                                "type": "ai_transcript_delta",
                                "delta": sc.output_transcription.text
                            })

                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                # Extract real-time speech subtitles from text part if provided
                                if part.text:
                                    logger.info(f"AI Subtitle (model_turn.part.text): {part.text}")
                                    await client_ws.send_json({
                                        "type": "ai_transcript_delta",
                                        "delta": part.text
                                    })
                                # Audio chunk (24kHz PCM)
                                if part.inline_data and part.inline_data.data:
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                                    await client_ws.send_json({
                                        "type": "audio_delta",
                                        "delta": audio_b64
                                    })

                        if sc.turn_complete:
                            logger.info("Turn complete from Gemini. Session remains open and listening.")
                            await client_ws.send_json({"type": "turn_complete"})

                except (asyncio.CancelledError, WebSocketDisconnect):
                    pass
                except Exception as e:
                    logger.error(f"Error in gemini-to-client stream: {e}")

            # Spawn both streaming background tasks (listener active before greeting)
            t1 = asyncio.create_task(forward_client_to_gemini())
            t2 = asyncio.create_task(forward_gemini_to_client())

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

            # Notify browser client that bridge is active and ready for caller speech
            await client_ws.send_json({"type": "ready"})

            # Wait until either client disconnects or Gemini closes
            done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            if 't1' in locals() and t1 and not t1.done():
                t1.cancel()
            if 't2' in locals() and t2 and not t2.done():
                t2.cancel()
            if session and hasattr(session, '_ws') and session._ws:
                try:
                    await session._ws.close()
                except Exception:
                    pass
            if connected_cm:
                try:
                    await connected_cm.__aexit__(None, None, None)
                    logger.info("Gemini Live session closed cleanly.")
                except Exception as ce:
                    logger.warning(f"Notice while closing Gemini session: {ce}")
            try:
                await gemini_client.aio.aclose()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Telephony bridge failed: {e}")
        try:
            await client_ws.send_json({
                "type": "error",
                "message": f"Connection to Gemini voice service failed: {str(e)}"
            })
        except Exception:
            pass
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
