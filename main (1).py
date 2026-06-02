"""
ScholarLens API — main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from src.api.routers import ingest, search, graph, trends
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("🚀 ScholarLens API starting up...")
    # Initialize FAISS index, DB tables, etc.
    from src.pipeline.embedder import EmbeddingIndex
    app.state.index = EmbeddingIndex()
    await app.state.index.initialize()
    logger.info("✅ Embedding index ready")
    yield
    logger.info("🛑 ScholarLens shutting down...")


app = FastAPI(
    title="ScholarLens API",
    description="AI-powered research paper analysis and knowledge graph builder",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow the React frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(graph.router, prefix="/graph", tags=["Knowledge Graph"])
app.include_router(trends.router, prefix="/trends", tags=["Trends"])


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "ScholarLens API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
