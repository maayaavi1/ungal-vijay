import os
import json
import asyncio
import logging
import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
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
logger = logging.getLogger("ungal_vijay_v2")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
if SUPABASE_URL.endswith("/rest/v1"):
    SUPABASE_URL = SUPABASE_URL[:-8].rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.0-flash-exp").strip()
GEMINI_LIVE_VOICE = os.getenv("GEMINI_LIVE_VOICE", "Kore").strip()
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
            logger.info(f"Loaded system prompt from: {p}")
            break
        except Exception as e:
            logger.warning(f"Could not load system prompt from {p}: {e}")

if not SYSTEM_PROMPT:
    SYSTEM_PROMPT = "You are the Tamil Nadu Government Welfare Schemes Voice AI helpline counselor. Speak in simple, respectful spoken Tamil with everyday public administrative nouns."

OPENING_GREETING = (
    "Vanakkam! ithu Tamil Nadu Arasu Thittangal helpline. Ungalukku enna thagaval venum sollunga?"
)

# Initialize Google GenAI client safely
ai_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(api_version="v1alpha"))
        logger.info("Google GenAI client initialized with API key (v1alpha).")
    except Exception as e:
        logger.error(f"Failed to initialize Google GenAI client: {e}")
else:
    logger.warning("GEMINI_API_KEY is not set in environment. It will be required when generating tokens.")

# ---------------------------------------------------------------------------
# 2. Welfare Scheme Fallback Data (Guarantees zero-failure cold boot)
# ---------------------------------------------------------------------------
FALLBACK_SCHEMES: List[Dict[str, Any]] = [
    {
        "id": "kmut-001",
        "name_en": "Kalaignar Magalir Urimai Thogai Thittam (KMUT)",
        "name_ta": "கலைஞர் மகளிர் உரிமைத் திட்டம்",
        "brief_benefit_en": "₹1,000 monthly financial assistance directly to women heads of family",
        "benefits_en": "₹1,000 per month deposited directly into beneficiary bank account via DBT.",
        "gender": ["female"],
        "min_age": 21,
        "max_age": 100,
        "marital_status": ["married", "widow", "deserted", "single"],
        "max_income": 250000,
        "community": ["all"],
        "application_procedure_en": "Apply through special camps held at local ration shops / e-Sevai centers with Ration card, Aadhaar card, and Bank passbook.",
        "required_documents_ta": "குடும்ப அட்டை (Ration Card), ஆதார் அட்டை, வங்கி கணக்கு புத்தகம், மின் கட்டண ரசீது",
        "required_documents_en": "Ration Card (Smart Card), Aadhaar Card, Bank Passbook linked to Aadhaar, Electricity Consumer Bill",
        "portal_url": "kmut.tn.gov.in",
        "helpline": "044-25619200",
        "officer_contact_ta": "மண்டல வருவாய் அலுவலர் / வட்டாட்சியர் (Tahsildar)",
        "special_conditions": ["women_head_of_family"]
    },
    {
        "id": "pudhumai-penn-002",
        "name_en": "Moovalur Ramamirtham Ammaiyar Higher Education Assurance Scheme (Pudhumai Penn)",
        "name_ta": "புதுமைப் பெண் திட்டம்",
        "brief_benefit_en": "₹1,000 monthly allowance for girl students pursuing higher education who studied in government schools",
        "benefits_en": "₹1,000 per month paid until completion of degree/diploma.",
        "gender": ["female"],
        "min_age": 17,
        "max_age": 25,
        "marital_status": ["single", "married"],
        "max_income": None,
        "community": ["all"],
        "application_procedure_en": "Apply online through your college nodal officer at penkalvi.tn.gov.in with Bonafide certificate and 6th-12th Govt school transfer certificate.",
        "required_documents_ta": "6 முதல் 12 ஆம் வகுப்பு வரை அரசுப் பள்ளியில் படித்ததற்கான சான்றிதழ், கல்லூரி சேர்க்கை அட்டை, ஆதார், வங்கி கணக்கு",
        "required_documents_en": "Govt school study proof (Class 6-12), College ID / Bonafide, Aadhaar Card, Bank Passbook",
        "portal_url": "penkalvi.tn.gov.in",
        "helpline": "181 / 044-25619200",
        "officer_contact_ta": "கல்லூரி திட்ட ஒருங்கிணைப்பாளர் / மாவட்ட சமூக நல அலுவலர்",
        "special_conditions": ["govt_school_studied", "higher_education"]
    },
    {
        "id": "tamil-pudhalvan-003",
        "name_en": "Tamil Pudhalvan Higher Education Scheme",
        "name_ta": "தமிழ்ப் புதல்வன் திட்டம்",
        "brief_benefit_en": "₹1,000 monthly allowance for boy students pursuing higher education who studied in government schools",
        "benefits_en": "₹1,000 per month disbursed to boys studying in college.",
        "gender": ["male"],
        "min_age": 17,
        "max_age": 25,
        "marital_status": ["single"],
        "max_income": None,
        "community": ["all"],
        "application_procedure_en": "Apply online via college nodal officer at tamilpudhalvan.tn.gov.in portal.",
        "required_documents_ta": "6-12 அரசு பள்ளி கல்வி சான்று, கல்லூரி அடையாள அட்டை, ஆதார், வங்கி கணக்கு புத்தகம்",
        "required_documents_en": "Govt school certificate (Class 6-12), College Bonafide/ID, Aadhaar Card, Bank Account",
        "portal_url": "tamilpudhalvan.tn.gov.in",
        "helpline": "181",
        "officer_contact_ta": "கல்லூரி முதல்வர் / மாவட்ட சமூக நல அலுவலர்",
        "special_conditions": ["govt_school_studied", "higher_education"]
    },
    {
        "id": "marriage-mra-004",
        "name_en": "Moovalur Ramamirtham Ammaiyar Memorial Marriage Assistance Scheme",
        "name_ta": "மூவலூர் ராமாமிர்தம் அம்மையார் நினைவு திருமண உதவி திட்டம்",
        "brief_benefit_en": "₹25,000 to ₹50,000 financial assistance plus 8 grams (1 sovereign) 22k gold coin for bride's marriage",
        "benefits_en": "Scheme 1: ₹25,000 cheque + 8g gold coin for 10th pass. Scheme 2: ₹50,000 cheque + 8g gold coin for degree/diploma holders.",
        "gender": ["female"],
        "min_age": 18,
        "max_age": 35,
        "marital_status": ["single"],
        "max_income": 72000,
        "community": ["all"],
        "application_procedure_en": "Apply through e-Sevai center at least 40 days before marriage date.",
        "required_documents_ta": "மணமகள் கல்வி சான்றிதழ், வருமான சான்று (₹72,000க்குள்), திருமண அழைப்பிதழ், மணமகன் & மணமகள் வயது சான்று",
        "required_documents_en": "Bride education mark sheet, Income certificate, Marriage invitation, Age proof of bride (18+) and groom (21+)",
        "portal_url": "tnesevai.tn.gov.in",
        "helpline": "181",
        "officer_contact_ta": "வட்டார வளர்ச்சி அலுவலர் (BDO) / மாவட்ட சமூக நல அலுவலர்",
        "special_conditions": ["marriage_assistance"]
    },
    {
        "id": "sewing-machine-005",
        "name_en": "Sathiyavani Muthu Ammaiyar Free Supply of Sewing Machine Scheme",
        "name_ta": "சத்தியவாணி முத்து அம்மையார் நினைவு இலவச தையல் இயந்திரம் வழங்கும் திட்டம்",
        "brief_benefit_en": "Free motorized/manual sewing machine for destitute widows, deserted wives, and poor women",
        "benefits_en": "Free sewing machine provided for self-employment livelihood.",
        "gender": ["female"],
        "min_age": 20,
        "max_age": 40,
        "marital_status": ["widow", "deserted", "destitute", "married"],
        "max_income": 72000,
        "community": ["all"],
        "application_procedure_en": "Submit application to District Social Welfare Office or nearest e-Sevai center with tailoring training certificate.",
        "required_documents_ta": "தையல் பயிற்சி சான்றிதழ் (குறைந்தது 6 மாதங்கள்), வருமான சான்றிதழ், ஆதரவற்ற விதவை / கணவனால் கைவிடப்பட்டோர் சான்றிதழ்",
        "required_documents_en": "Tailoring training certificate (minimum 6 months), Income certificate (below ₹72,000), Widow / Deserted certificate if applicable",
        "portal_url": "tnesevai.tn.gov.in",
        "helpline": "181",
        "officer_contact_ta": "மாவட்ட சமூக நல அலுவலர் (DSWO) / தாலுகா சமூக நல விரிவாக்க அலுவலர்",
        "special_conditions": ["tailoring_trained"]
    },
    {
        "id": "goat-sheep-006",
        "name_en": "Free Distribution of Goats / Sheep Scheme",
        "name_ta": "இலவச வெள்ளாடுகள் / செம்மறியாடுகள் வழங்கும் திட்டம்",
        "brief_benefit_en": "4 free goats/sheep per poorest rural landless family with 100% subsidy and free fodder seeds",
        "benefits_en": "4 goats or sheep (1 male + 3 female) given free of cost for sustainable rural livelihood.",
        "gender": ["female", "male"],
        "min_age": 18,
        "max_age": 60,
        "marital_status": ["all"],
        "max_income": 72000,
        "community": ["all", "sc", "st", "bc", "mbc"],
        "application_procedure_en": "Beneficiaries selected by Grama Sabha convened by Village Panchayat.",
        "required_documents_ta": "குடும்ப அட்டை, ஆதார் அட்டை, நிலமற்ற விவசாய கூலி சான்றிதழ், கிராம சபை ஒப்புதல்",
        "required_documents_en": "Ration Card, Aadhaar Card, Landless agricultural labourer proof from VAO",
        "portal_url": "ahvs.tn.gov.in",
        "helpline": "1962",
        "officer_contact_ta": "மண்டல கால்நடை உதவி இயக்குனர் / கால்நடை மருந்தக உதவி மருத்துவர்",
        "special_conditions": ["landless_agricultural_labourer", "rural"]
    }
]

# ---------------------------------------------------------------------------
# 3. Database Repository (Supabase with In-Memory Cache)
# ---------------------------------------------------------------------------
class SchemeRepository:
    def __init__(self, url: str, key: str):
        self.client: Optional[Client] = None
        if url and key:
            try:
                self.client = create_client(url, key)
                logger.info("Supabase client created successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
        else:
            logger.warning("Supabase credentials missing. Using embedded fallback schemes.")

        self.cached_schemes: List[Dict[str, Any]] = list(FALLBACK_SCHEMES)

    async def warm_up_cache(self):
        """Preloads all 25 schemes from Supabase into memory."""
        if not self.client:
            return
        try:
            response = await asyncio.to_thread(
                lambda: self.client.table("schemes").select("*").execute()
            )
            if response.data and len(response.data) > 0:
                self.cached_schemes = response.data
                logger.info(f"Supabase cache initialized: {len(self.cached_schemes)} schemes loaded.")
            else:
                logger.info("Supabase query returned empty. Keeping fallback schemes.")
        except Exception as e:
            logger.error(f"Error fetching schemes from Supabase during warmup: {e}. Keeping fallback schemes.")

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
        results = self.cached_schemes

        # 1. Keyword search with stop-word filtering
        if query and isinstance(query, str) and query.strip():
            raw_terms = query.lower().strip().split()
            STOP_WORDS = {
                "scheme", "schemes", "thittam", "thittangal", "for", "the", "a", "an",
                "in", "of", "and", "to", "arasu", "government", "tamil", "nadu", "service",
                "pathi", "solla", "sollunga", "irukka", "kedaikkuma", "details", "enna",
                "sevai", "sevai-kku", "help", "helpline", "scheme-kku", "scheme-gal", "about"
            }
            meaningful_terms = [t for t in raw_terms if t not in STOP_WORDS and len(t) > 2]

            if meaningful_terms:
                scored = []
                for s in results:
                    score = 0
                    name_text = f"{s.get('name_en') or ''} {s.get('name_ta') or ''}".lower()
                    benefit_text = f"{s.get('brief_benefit_en') or ''} {' '.join(s.get('special_conditions') or [])}".lower()
                    body_text = f"{s.get('objective_en') or ''} {s.get('benefits_en') or ''}".lower()

                    for t in meaningful_terms:
                        if t in name_text:
                            score += 5
                        elif t in benefit_text:
                            score += 3
                        elif t in body_text:
                            score += 1

                    if score > 0:
                        scored.append((score, s))

                scored.sort(key=lambda x: x[0], reverse=True)
                results = [s for score, s in scored]

        # Exact category token matching helper (prevents 'male' in 'female' or 'bc' in 'mbc')
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
            return bool(targets & categories)

        # 2. Gender filtering
        if gender:
            results = [s for s in results if match_tokens(s.get("gender"), gender)]

        # 3. Age filtering
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

        # 4. Income filtering
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

        # 5. Marital Status filtering
        if marital_status:
            results = [s for s in results if match_tokens(s.get("marital_status"), marital_status)]

        # 6. Community filtering
        if community:
            results = [s for s in results if match_tokens(s.get("community"), community)]

        # 7. Special condition filtering
        if special_condition:
            cond_val = str(special_condition).lower()
            results = [
                s for s in results
                if s.get("special_conditions") and cond_val in [c.lower() for c in s["special_conditions"]]
            ]

        return results

    def get_scheme_by_id_or_name(self, identifier: str) -> Optional[Dict[str, Any]]:
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
# 4. Tool Declarations (Clean JSON Schema for Gemini Live API)
# ---------------------------------------------------------------------------
SEARCH_SCHEMES_DECLARATION = {
    "name": "search_schemes",
    "description": (
        "Searches Tamil Nadu welfare schemes using caller keywords or the Big 4 triage fields "
        "(age, gender, marital status, income, community). Call this when the caller mentions a specific need, "
        "a scheme name, or answers triage questions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Scheme name keyword or benefit keyword, e.g. 'marriage', 'pudhumai penn', 'goat', 'sewing machine', 'loan'."
            },
            "age": {
                "type": "integer",
                "description": "Age of the beneficiary."
            },
            "gender": {
                "type": "string",
                "description": "Gender of the beneficiary: 'female', 'male', 'transgender'."
            },
            "marital_status": {
                "type": "string",
                "description": "Marital/family status of beneficiary: 'single', 'married', 'widow', 'deserted', 'destitute'."
            },
            "income": {
                "type": "integer",
                "description": "Annual family income in Rupees, e.g. 72000 or 250000."
            },
            "community": {
                "type": "string",
                "description": "Caste category if relevant: 'sc', 'st', 'scc', 'bc', 'mbc', 'fc'."
            },
            "special_condition": {
                "type": "string",
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
        "type": "object",
        "properties": {
            "scheme_identifier": {
                "type": "string",
                "description": "Scheme UUID or scheme English/Tamil name."
            }
        },
        "required": ["scheme_identifier"]
    }
}

async def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Executes tool lookups against in-memory scheme cache."""
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

        # Mode 1: Single Match -> 1-liner benefit + preloaded details
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

        # Mode 2: Multiple Matches -> 2 to 4 concise options
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
# 5. FastAPI App Lifecycle & Routes
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Ungal Vijay v2 Backend...")
    await db_repo.warm_up_cache()
    yield
    logger.info("Ungal Vijay v2 Backend shutdown.")

app = FastAPI(title="Ungal Vijay v2 (Gemini Live Solved Architecture)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "v3.0",
        "cached_schemes": len(db_repo.cached_schemes),
        "model": GEMINI_LIVE_MODEL,
        "voice": GEMINI_LIVE_VOICE
    }

@app.post("/api/token")
async def create_ephemeral_token():
    """Generates an ephemeral token for direct browser-to-Gemini WebSocket connection."""
    global ai_client
    if not ai_client:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            ai_client = genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1alpha"))
        else:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY is not configured on the backend server. Please set GEMINI_API_KEY in environment variables."
            )

    try:
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        expire_time = now + datetime.timedelta(minutes=30)
        
        token = await asyncio.to_thread(
            lambda: ai_client.auth_tokens.create(
                config={
                    "uses": 1,
                    "expire_time": expire_time.isoformat(),
                    "new_session_expire_time": (now + datetime.timedelta(minutes=5)).isoformat(),
                    "http_options": {"api_version": "v1alpha"},
                }
            )
        )
        return {
            "token": token.name,
            "expires_at": expire_time.isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating ephemeral token: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate ephemeral token: {str(e)}")

@app.get("/api/config")
async def get_client_config():
    """Returns runtime model configuration and tool schemas for client setup."""
    return {
        "model": GEMINI_LIVE_MODEL,
        "voice": GEMINI_LIVE_VOICE,
        "system_instruction": SYSTEM_PROMPT,
        "tools": [SEARCH_SCHEMES_DECLARATION, GET_SCHEME_DETAILS_DECLARATION],
        "greeting": OPENING_GREETING
    }

@app.post("/api/search_schemes")
async def api_search_schemes(request: Request):
    """Backend execution endpoint for Gemini function call: search_schemes."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    result = await execute_tool("search_schemes", body)
    return result

@app.post("/api/get_scheme_details")
async def api_get_scheme_details(request: Request):
    """Backend execution endpoint for Gemini function call: get_scheme_details."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    result = await execute_tool("get_scheme_details", body)
    return result

# ---------------------------------------------------------------------------
# 6. Static Files & Frontend Serving
# ---------------------------------------------------------------------------
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
    logger.info(f"Serving frontend assets from: {frontend_dir}")
    
    # Mount subdirectories if present
    img_dir = os.path.join(frontend_dir, "img")
    if os.path.exists(img_dir):
        app.mount("/img", StaticFiles(directory=img_dir), name="img")
        app.mount("/frontend/img", StaticFiles(directory=img_dir), name="frontend_img")

    processors_dir = os.path.join(frontend_dir, "audio-processors")
    if os.path.exists(processors_dir):
        app.mount("/audio-processors", StaticFiles(directory=processors_dir), name="processors")
        app.mount("/frontend/audio-processors", StaticFiles(directory=processors_dir), name="frontend_processors")

    @app.get("/")
    async def serve_index():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse({"message": "Frontend index.html not found"}, status_code=404)

    # Mount remaining static files at root
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend_root")
else:
    logger.warning("No frontend directory found. Serving API endpoints only.")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Ungal Vijay v3 on http://{HOST}:{PORT}")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
