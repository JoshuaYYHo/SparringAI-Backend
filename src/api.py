"""
Sparring AI — Video Upload API
Upload a boxing video → ML analysis (main.py) → OpenAI GPT-4o fight commentary.
"""

import os
import json
import uuid
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from openai import OpenAI

from src.main import identify_punches_in_video

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

SPARRING_API_KEY = os.getenv("SPARRING_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_raw_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins if o.strip()]
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# OpenAI client (initialized once)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Allowed video MIME types and extensions
ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",      # .mov
    "video/x-msvideo",      # .avi
    "video/webm",
    "application/octet-stream",  # curl / mobile clients often send this
}
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}

# Magic bytes for video container validation
VIDEO_MAGIC_BYTES = {
    b"\x00\x00\x00":  "ftyp/mp4/mov",
    b"\x1a\x45\xdf":  "webm/mkv",
    b"\x52\x49\x46":  "avi",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sparring_api")

# ---------------------------------------------------------------------------
# ✏️  AI PROMPT — EDIT THIS TO CUSTOMIZE THE AI ANALYSIS
# ---------------------------------------------------------------------------
# This is the system prompt that tells GPT-4o HOW to analyze the fight data.
# The ML pipeline JSON will be injected into the user message automatically.
# Modify this to change the tone, focus, or structure of the analysis.

SYSTEM_PROMPT = """
You are an expert boxing coach and fight analyst. You will receive structured 
JSON data from an AI vision system that tracked a sparring session. The data 
contains detected punches, defensive moves, stance information, and analytics 
for each fighter.

Your job is to provide a comprehensive, actionable fight analysis. Include:

1. **Fight Summary** — A brief overview of what happened in the session.
2. **Punch Breakdown** — How many of each punch type each fighter threw, 
   who was more active, and any notable patterns.
3. **Defensive Assessment** — How well each fighter defended (slips, ducks, 
   blocks) relative to the volume of punches thrown at them.
4. **Stance & Footwork** — Comment on stance consistency, stance width, 
   and which fighter controlled distance.
5. **Areas for Improvement** — Specific, actionable tips for each fighter 
   based on the data.
6. **Score** — Give each fighter a 0-100 performance score with justification.

Keep the language conversational but professional, like a real boxing coach 
giving feedback after watching a round. Use the actual numbers from the data 
to support your points.
"""

# This is the user message template. {analysis_json} will be replaced with
# the actual ML pipeline output. You can add extra instructions here too.

USER_PROMPT_TEMPLATE = """
Here is the fight analysis data from the AI vision system:

```json
{analysis_json}
```

Please provide a detailed fight analysis based on this data.
"""

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Sparring AI API",
    description="Upload boxing videos for AI-powered fight analysis.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _verify_api_key(api_key: Optional[str]) -> None:
    """Reject the request if the API key is missing or wrong."""
    if not SPARRING_API_KEY:
        logger.error("SPARRING_API_KEY is not set in .env")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error. Contact support.",
        )
    if not api_key or api_key != SPARRING_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


def _validate_file(filename: str, content_type: str, header_bytes: bytes) -> None:
    """
    Three-layer file validation:
      1. MIME type from the upload metadata
      2. File extension
      3. Magic bytes (file header signature)
    """
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{content_type}'. Allowed: {', '.join(ALLOWED_MIME_TYPES)}",
        )

    ext = os.path.splitext(filename)[1].lower() if filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    is_valid_magic = False
    for magic in VIDEO_MAGIC_BYTES:
        if header_bytes[:len(magic)] == magic:
            is_valid_magic = True
            break
    if not is_valid_magic:
        raise HTTPException(
            status_code=400,
            detail="File content does not match any known video format. Upload rejected.",
        )



def _summarize_for_llm(analysis_data: dict) -> dict:
    """
    Compresses the raw ML pipeline output into a compact summary for the LLM.
    The raw data can be 800KB+ (171K tokens) due to per-frame distance analytics.
    This brings it down to ~2-5KB which fits comfortably within GPT-4o's limits.
    """
    summary = {}

    # --- Summarize per-fighter punch data ---
    for role in ["you", "opponent"]:
        if role in analysis_data:
            fighter = analysis_data[role]
            punches = fighter.get("punches_identified", [])
            defenses = fighter.get("defensive_moves", [])

            # Count punches by type
            punch_counts = {}
            stance_counts = {}
            analytics_agg = {"stance_width": [], "avg_guard_distance": [], "torso_angle": [], "elbow_bend": []}

            for p in punches:
                ptype = p["punch_type"]
                punch_counts[ptype] = punch_counts.get(ptype, 0) + 1
                a = p.get("analytics", {})
                stance_counts[a.get("stance", "Unknown")] = stance_counts.get(a.get("stance", "Unknown"), 0) + 1
                if "stance_width" in a: analytics_agg["stance_width"].append(a["stance_width"])
                if "avg_guard_distance" in a: analytics_agg["avg_guard_distance"].append(a["avg_guard_distance"])
                if "torso_angle_degrees" in a: analytics_agg["torso_angle"].append(a["torso_angle_degrees"])
                if "active_elbow_bend" in a: analytics_agg["elbow_bend"].append(a["active_elbow_bend"])

            # Defense counts by type
            defense_counts = {}
            for d in defenses:
                dtype = d["punch_type"]
                defense_counts[dtype] = defense_counts.get(dtype, 0) + 1

            # Build averages
            avg_stats = {}
            for key, vals in analytics_agg.items():
                if vals:
                    avg_stats[f"avg_{key}"] = round(sum(vals) / len(vals), 2)

            summary[role] = {
                "appearance": fighter.get("appearance", {}),
                "total_punches": len(punches),
                "punch_breakdown": punch_counts,
                "total_defensive_moves": len(defenses),
                "defense_breakdown": defense_counts,
                "dominant_stance": max(stance_counts, key=stance_counts.get) if stance_counts else "Unknown",
                "stance_distribution": stance_counts,
                **avg_stats,
            }

    # --- Fallback for non-personalized mode (punches_identified at top level) ---
    if "punches_identified" in analysis_data:
        punches = analysis_data["punches_identified"]
        by_fighter = {}
        for p in punches:
            fid = p.get("fighter_id", 0)
            if fid not in by_fighter:
                by_fighter[fid] = {"punch_counts": {}, "total": 0}
            by_fighter[fid]["total"] += 1
            ptype = p["punch_type"]
            by_fighter[fid]["punch_counts"][ptype] = by_fighter[fid]["punch_counts"].get(ptype, 0) + 1
        summary["fighters"] = by_fighter

    # --- Include fighter appearances for non-personalized mode ---
    if "fighter_appearances" in analysis_data:
        summary["fighter_appearances"] = analysis_data["fighter_appearances"]

    # --- Summarize distance analytics (sample every 10th entry) ---
    frame_analytics = analysis_data.get("global_fight_analytics", [])
    if frame_analytics:
        distances = [f["distance_between_fighters_px"] for f in frame_analytics if "distance_between_fighters_px" in f]
        closers = [f.get("distance_closed_by") for f in frame_analytics if "distance_closed_by" in f]

        closer_counts = {}
        for c in closers:
            if c:
                closer_counts[c] = closer_counts.get(c, 0) + 1

        summary["ring_analytics"] = {
            "total_frames_tracked": len(frame_analytics),
            "avg_distance_px": round(sum(distances) / len(distances), 1) if distances else 0,
            "min_distance_px": round(min(distances), 1) if distances else 0,
            "max_distance_px": round(max(distances), 1) if distances else 0,
            "distance_closed_by_counts": closer_counts,
        }

    return summary


def _generate_ai_analysis(analysis_data: dict) -> str:
    """
    Sends the ML pipeline JSON to OpenAI GPT-4o and returns the written analysis.
    Uses the prompts defined at the top of this file.
    """
    if openai_client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set in .env. Cannot generate AI analysis.",
        )

    # Compress raw data to fit within GPT-4o token limits
    summarized = _summarize_for_llm(analysis_data)
    analysis_json_str = json.dumps(summarized, indent=2)
    logger.info(f"Summarized analysis data: {len(analysis_json_str)} chars (from ~800KB raw)")

    user_message = USER_PROMPT_TEMPLATE.format(
        analysis_json=analysis_json_str
    )

    logger.info("Sending analysis data to OpenAI GPT-4o...")

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_message.strip()},
        ],
        temperature=0.7,
        max_tokens=4096,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the Sparring AI Backend!"}

@app.get("/health")
async def health_check():
    """Public health-check endpoint (no auth required)."""
    return {"status": "ok"}

@app.post("/video_upload")
@limiter.limit("1/minute")
async def video_upload(
    request: Request,
    video: UploadFile = File(..., description="The boxing video file to analyse."),
    confidence_threshold: float = Query(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to register a punch (0.0–1.0).",
    ),
    user_bbox: Optional[str] = Query(
        default=None,
        description="Bounding box to identify the user in frame 1, format: x1,y1,x2,y2",
    ),
    x_api_key: Optional[str] = Header(None),
):
    """
    Upload a boxing video for AI analysis.

    **Pipeline:**
    1. Validates the upload (auth, file type, size)
    2. Runs the ML vision pipeline from main.py → punch/defense JSON
    3. Feeds the JSON into OpenAI GPT-4o → written fight analysis

    **Returns:**
    - `raw_analysis`: The structured JSON from the ML pipeline
    - `ai_commentary`: GPT-4o's written fight analysis
    """

    # ── Auth ──────────────────────────────────────────────────────────────
    _verify_api_key(x_api_key)

    # ── Read file header for magic-byte validation ────────────────────────
    header_bytes = await video.read(12)
    if len(header_bytes) < 3:
        raise HTTPException(status_code=400, detail="Uploaded file is too small to be a video.")
    await video.seek(0)

    # ── Validate type / extension / magic bytes ───────────────────────────
    _validate_file(video.filename, video.content_type, header_bytes)

    # ── Parse optional user_bbox ──────────────────────────────────────────
    parsed_bbox = None
    if user_bbox:
        try:
            parts = [float(x.strip()) for x in user_bbox.split(",")]
            if len(parts) != 4:
                raise ValueError
            parsed_bbox = parts
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="user_bbox must be in the format x1,y1,x2,y2 (e.g. 100,200,300,400).",
            )

    # ── Save to temp file & run full pipeline ─────────────────────────────
    temp_path = None
    try:
        ext = os.path.splitext(video.filename)[1].lower() if video.filename else ".mp4"
        temp_path = os.path.join(tempfile.gettempdir(), f"sparring_{uuid.uuid4().hex}{ext}")

        total_bytes = 0
        with open(temp_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE_MB} MB.",
                    )
                f.write(chunk)

        logger.info(f"Received video ({total_bytes / (1024*1024):.1f} MB). Starting ML analysis…")

        # ── Step 1: Run the ML pipeline from main.py ──────────────────────
        raw_analysis = identify_punches_in_video(
            video_path=temp_path,
            confidence_threshold=confidence_threshold,
            user_bbox=parsed_bbox,
        )

        logger.info("ML analysis complete. Generating GPT-4o commentary…")

        # ── Step 2: Feed JSON into OpenAI for written analysis ───────────
        ai_commentary = _generate_ai_analysis(raw_analysis)

        logger.info("GPT-4o analysis complete.")

        return JSONResponse(content={
            "success": True,
            "raw_analysis": raw_analysis,
            "ai_commentary": ai_commentary,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again or contact support.")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info("Temp file cleaned up.")
