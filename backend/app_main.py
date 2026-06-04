"""
app_main.py - FastAPI Web Application
======================================
The main entry point for the HR Policy Chatbot API server.

This file:
  - Creates the FastAPI app
  - Sets up CORS (so a website can call our API)
  - Initializes all services (Google Drive, document processor, RAG pipeline)
  - Defines all API endpoints
  - Starts background tasks

API ENDPOINTS:
  GET  /                          - Welcome message
  GET  /health                    - Check if server is running
  POST /api/v1/sync/sync-documents - Pull latest docs from Google Drive
  POST /api/v1/chat/query          - Ask a question about HR policies
  GET  /api/v1/chat/stats          - See how many documents are loaded

HOW TO RUN:
  uvicorn app_main:app --reload --host 0.0.0.0 --port 8000

THEN VISIT:
  http://localhost:8000       - Welcome message
  http://localhost:8000/docs  - Interactive API documentation (auto-generated!)
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import settings, print_config_summary
from google_drive_service import GoogleDriveService
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline

# =============================================================================
# LOGGING SETUP
# =============================================================================
# Configure logging to show timestamp, level, and message
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL SERVICE INSTANCES
# These are created once when the server starts and shared across all requests
# =============================================================================
drive_service = GoogleDriveService()
doc_processor = DocumentProcessor()
rag_pipeline = RAGPipeline()

# Track sync status
sync_status = {
    "is_syncing": False,
    "last_sync_time": None,
    "last_sync_result": None,
    "last_error": None,
}

# How often to auto-sync in the background (seconds). Default: 30 minutes.
AUTO_SYNC_INTERVAL_SECONDS = 30 * 60


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

async def _periodic_sync_loop():
    """
    Background coroutine that automatically syncs Google Drive every
    AUTO_SYNC_INTERVAL_SECONDS (default: 30 minutes).

    Runs forever until the server shuts down.
    First sync happens immediately at startup, then repeats on the interval.
    """
    while True:
        try:
            logger.info("Auto-sync: starting scheduled sync from Google Drive...")
            run_document_sync()
        except Exception as e:
            logger.error(f"Auto-sync error: {e}")

        logger.info(
            f"Auto-sync: next sync in {AUTO_SYNC_INTERVAL_SECONDS // 60} minutes."
        )
        await asyncio.sleep(AUTO_SYNC_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on server START and STOP.

    Startup sequence:
      1. Connect to OpenAI
      2. Immediately sync documents from Google Drive
      3. Launch background loop to re-sync every 30 minutes automatically
    """
    # ---- STARTUP ----
    logger.info("=" * 60)
    logger.info(f"  Starting {settings.app_name} v{settings.app_version}")
    logger.info("=" * 60)
    print_config_summary()

    # Step 1: Connect to OpenAI
    if rag_pipeline.initialize():
        logger.info("RAG pipeline initialized successfully.")
    else:
        logger.warning(
            "RAG pipeline failed to initialize. "
            "Check OPENAI_API_KEY in .env file."
        )

    # Step 2: Launch background auto-sync loop
    # This syncs immediately on startup, then every 30 minutes automatically.
    sync_task = asyncio.create_task(_periodic_sync_loop())
    logger.info(
        f"Auto-sync enabled: syncing now and every "
        f"{AUTO_SYNC_INTERVAL_SECONDS // 60} minutes automatically."
    )

    logger.info("Server is ready! Visit http://localhost:8005 to open the chatbot.")

    yield  # ← server runs here

    # ---- SHUTDOWN ----
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass
    logger.info("Server shut down cleanly.")


# =============================================================================
# CREATE THE FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
    # The /docs URL gives you an interactive API browser - very useful!
    docs_url="/docs",
    redoc_url="/redoc",
)

# =============================================================================
# CORS MIDDLEWARE
# =============================================================================
# CORS (Cross-Origin Resource Sharing) lets websites from other domains
# call our API. During development, we allow all origins (*).
# In production, you should list specific allowed origins.

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# These define what data the API accepts and returns (with validation)
# =============================================================================

class ChatQueryRequest(BaseModel):
    """Request body for the /chat/query endpoint."""
    question: str = Field(
        ...,  # ... means required
        min_length=3,
        max_length=1000,
        description="The HR policy question to answer",
        example="How many vacation days do new employees get?",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of document chunks to use (1-20). Defaults to settings.",
    )


class ChatQueryResponse(BaseModel):
    """Response from the /chat/query endpoint."""
    answer: str = Field(description="The AI-generated answer")
    sources: list[dict] = Field(description="Which documents were used")
    confidence_score: float = Field(description="How confident we are (0.0-1.0)")
    query: str = Field(description="The original question")
    model_used: str = Field(description="Which GPT model answered")
    tokens_used: int = Field(description="OpenAI tokens consumed")
    retrieved_chunks: int = Field(description="Number of document chunks retrieved")
    processing_time_seconds: float = Field(description="How long the query took")


class SyncResponse(BaseModel):
    """Response from the /sync/sync-documents endpoint."""
    message: str
    status: str
    documents_found: int = 0
    chunks_created: int = 0
    sync_time_seconds: float = 0.0


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    app_name: str
    version: str
    openai_configured: bool
    google_drive_authenticated: bool
    documents_loaded: int
    is_syncing: bool
    last_sync: Optional[str]


# =============================================================================
# BACKGROUND SYNC FUNCTION
# =============================================================================

def run_document_sync():
    """
    Background task: Download documents from Google Drive and embed them.

    This runs in the background so the API response returns immediately
    while the (possibly slow) sync continues in the background.
    """
    global sync_status

    if sync_status["is_syncing"]:
        logger.warning("Sync already in progress, skipping.")
        return

    sync_status["is_syncing"] = True
    sync_status["last_error"] = None
    start_time = time.time()

    logger.info("Background sync started...")

    try:
        # Step 1: Authenticate with Google Drive
        if not drive_service.is_authenticated:
            logger.info("Authenticating with Google Drive...")
            if not drive_service.authenticate():
                raise RuntimeError(
                    "Google Drive authentication failed. "
                    "Check credentials.json file."
                )

        # Step 2: Download all documents from Google Drive
        logger.info("Downloading documents from Google Drive...")
        downloaded = drive_service.sync_all_documents()

        if not downloaded:
            sync_status["last_sync_result"] = {
                "documents_found": 0,
                "chunks_created": 0,
                "message": "No documents found in the HR Policies folder.",
            }
            logger.warning("No documents found to sync.")
            return

        # Step 3: Process documents into chunks
        file_paths = [doc["path"] for doc in downloaded]
        logger.info(f"Processing {len(file_paths)} downloaded document(s)...")
        chunks = doc_processor.process_multiple_files(file_paths)

        # Step 4: Clear old documents and load new ones into RAG
        logger.info("Loading chunks into RAG pipeline...")
        rag_pipeline.clear_documents()
        chunks_added = rag_pipeline.add_documents(chunks)

        elapsed = time.time() - start_time
        sync_status["last_sync_result"] = {
            "documents_found": len(downloaded),
            "chunks_created": chunks_added,
            "sync_time_seconds": round(elapsed, 2),
            "message": "Sync completed successfully!",
        }
        sync_status["last_sync_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
            f"Sync complete! {len(downloaded)} documents, "
            f"{chunks_added} chunks loaded in {elapsed:.1f}s"
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Sync failed: {error_msg}")
        sync_status["last_error"] = error_msg
        sync_status["last_sync_result"] = {
            "error": error_msg,
            "message": "Sync failed. See logs for details.",
        }

    finally:
        sync_status["is_syncing"] = False


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Serve the chatbot UI."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {
        "message": f"Welcome to {settings.app_name}!",
        "version": settings.app_version,
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Check the health status of all services.

    Use this to verify:
    - Server is running
    - OpenAI is configured
    - Google Drive is authenticated
    - Documents are loaded
    """
    stats = rag_pipeline.get_stats()

    return HealthResponse(
        status="healthy" if rag_pipeline.is_ready else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        openai_configured=settings.is_openai_configured,
        google_drive_authenticated=drive_service.is_authenticated,
        documents_loaded=stats.get("total_chunks", 0),
        is_syncing=sync_status["is_syncing"],
        last_sync=sync_status.get("last_sync_time"),
    )


@app.post(
    "/api/v1/sync/sync-documents",
    response_model=SyncResponse,
    tags=["Document Sync"],
)
async def sync_documents(background_tasks: BackgroundTasks):
    """
    Start syncing HR policy documents from Google Drive.

    This starts a background task that:
    1. Downloads all documents from your Google Drive "HR Policies" folder
    2. Extracts and cleans the text
    3. Splits text into chunks
    4. Creates embeddings for each chunk
    5. Stores everything in memory for fast retrieval

    The response returns immediately. The sync continues in the background.
    Check /health or /api/v1/chat/stats to see when it's done.

    IMPORTANT: Run this endpoint after:
    - First setup
    - Adding new documents to Google Drive
    - Updating existing documents
    """
    if sync_status["is_syncing"]:
        return SyncResponse(
            message="Sync is already in progress. Please wait.",
            status="in_progress",
        )

    if not settings.is_openai_configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenAI API key not configured. "
                "Please set OPENAI_API_KEY in your .env file."
            ),
        )

    # Start the sync in the background
    background_tasks.add_task(run_document_sync)

    return SyncResponse(
        message=(
            "Sync started in the background! "
            "Check /health or /api/v1/chat/stats to see progress. "
            "Sync may take 1-5 minutes depending on number of documents."
        ),
        status="started",
    )


@app.post(
    "/api/v1/chat/query",
    response_model=ChatQueryResponse,
    tags=["Chat"],
)
async def query_hr_policy(request: ChatQueryRequest):
    """
    Ask a question about HR policies.

    The chatbot will:
    1. Find the most relevant sections from your HR documents
    2. Use GPT-4 to generate a precise answer based on those sections
    3. Return the answer with source citations and confidence score

    REQUIREMENTS:
    - Documents must be synced first (use /api/v1/sync/sync-documents)
    - OpenAI API key must be configured in .env

    EXAMPLE REQUEST:
    ```json
    {
        "question": "How many sick days do employees get per year?"
    }
    ```
    """
    if not rag_pipeline.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG pipeline is not ready. "
                "Check OPENAI_API_KEY in .env file and restart the server."
            ),
        )

    if not rag_pipeline.stored_documents:
        # If no docs loaded yet, trigger a sync automatically instead of erroring
        if not sync_status["is_syncing"]:
            logger.info("No documents loaded — triggering automatic sync before answering.")
            import threading
            threading.Thread(target=run_document_sync, daemon=True).start()
        raise HTTPException(
            status_code=503,
            detail=(
                "Documents are being loaded automatically. "
                "Please wait 1-2 minutes and try again."
            ),
        )

    start_time = time.time()

    # Override top_k if specified in request
    if request.top_k:
        rag_pipeline.top_k = request.top_k

    # Get the answer from the RAG pipeline
    response = rag_pipeline.answer_question(request.question)

    # Reset top_k to default
    rag_pipeline.top_k = settings.top_k_results

    processing_time = round(time.time() - start_time, 3)

    return ChatQueryResponse(
        answer=response.answer,
        sources=response.sources,
        confidence_score=response.confidence_score,
        query=response.query,
        model_used=response.model_used,
        tokens_used=response.tokens_used,
        retrieved_chunks=response.retrieved_chunks,
        processing_time_seconds=processing_time,
    )


@app.get("/api/v1/chat/stats", tags=["Chat"])
async def get_chat_stats():
    """
    Get statistics about the currently loaded documents.

    Shows:
    - How many documents are loaded
    - How many chunks per document
    - Estimated token counts
    - Whether the pipeline is ready
    """
    stats = rag_pipeline.get_stats()
    stats["sync_status"] = sync_status

    return stats


# =============================================================================
# MAIN ENTRY POINT - Run directly with Python
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(
        f"Starting server on http://{settings.host}:{settings.port}\n"
        f"API docs available at http://localhost:{settings.port}/docs"
    )

    uvicorn.run(
        "app_main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,  # Auto-reload when code changes (dev only)
        log_level="debug" if settings.debug else "info",
    )
