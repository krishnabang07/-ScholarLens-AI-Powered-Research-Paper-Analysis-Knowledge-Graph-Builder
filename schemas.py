"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Ingestion ────────────────────────────────────────────────────────────────

class ArxivIngestRequest(BaseModel):
    arxiv_id: str = Field(..., example="2310.06825")
    include_references: bool = Field(False, description="Also ingest cited papers (1 hop)")

class IngestResponse(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: Optional[int]
    entities_found: int
    graph_nodes_added: int
    already_exists: bool = False


# ── Search ───────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: Optional[int]
    score: float
    snippet: str
    arxiv_id: Optional[str]

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_papers_searched: int


# ── Knowledge Graph ──────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    label: str
    type: str          # "concept" | "method" | "dataset" | "author" | "paper"
    weight: float      # frequency / importance score
    papers: list[str]  # paper IDs mentioning this node
    x: Optional[float] = None
    y: Optional[float] = None

class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float      # co-occurrence count
    edge_type: str     # "co-occurs" | "cites" | "authored_by"

class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict


# ── Trends ───────────────────────────────────────────────────────────────────

class TrendDataPoint(BaseModel):
    year: int
    count: int
    notable_papers: list[str]

class TrendsResponse(BaseModel):
    topic: str
    data: list[TrendDataPoint]
    peak_year: int
    trend_direction: str   # "rising" | "falling" | "stable"


# ── Summarization ────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    concept: str
    max_papers: int = Field(20, ge=1, le=50)

class SummarizeResponse(BaseModel):
    concept: str
    summary: str
    papers_used: int
    sections: dict[str, str]  # intro, methods, findings, gaps
