"""
pipeline/ingestion.py — orchestrates the full ingestion pipeline.

Steps:
  1. Extract text (arXiv API or PDF)
  2. Run entity extraction
  3. Add to FAISS index (embeddings)
  4. Update knowledge graph
  5. Persist metadata to SQLite
"""

import uuid
import hashlib
from datetime import datetime

import arxiv
import fitz  # PyMuPDF

from src.api.schemas import IngestResponse
from src.pipeline.entity_extractor import EntityExtractor
from src.pipeline.embedder import EmbeddingIndex
from src.graph.builder import KnowledgeGraphBuilder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Singletons — initialized once per process
_extractor: EntityExtractor = None
_graph: KnowledgeGraphBuilder = None


def _get_extractor() -> EntityExtractor:
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def _get_graph() -> KnowledgeGraphBuilder:
    global _graph
    if _graph is None:
        _graph = KnowledgeGraphBuilder()
    return _graph


async def paper_exists(paper_id: str) -> bool:
    """Check if paper_id is already in the graph."""
    graph = _get_graph()
    return graph.G.has_node(paper_id)


class PaperIngestionPipeline:

    async def process_arxiv_paper(
        self,
        arxiv_paper: arxiv.Result,
        index: EmbeddingIndex,
        include_references: bool = False,
    ) -> IngestResponse:
        paper_id = f"arxiv::{arxiv_paper.entry_id.split('/')[-1]}"
        
        if await paper_exists(paper_id):
            return IngestResponse(
                paper_id=paper_id,
                title=arxiv_paper.title,
                authors=[a.name for a in arxiv_paper.authors],
                year=arxiv_paper.published.year if arxiv_paper.published else None,
                entities_found=0,
                graph_nodes_added=0,
                already_exists=True,
            )

        text = f"{arxiv_paper.title}\n\n{arxiv_paper.summary}"
        authors = [a.name for a in arxiv_paper.authors]
        year = arxiv_paper.published.year if arxiv_paper.published else datetime.now().year

        return await self._run_pipeline(paper_id, arxiv_paper.title, authors, year, text, index)

    async def process_pdf(
        self,
        pdf_path: str,
        filename: str,
        index: EmbeddingIndex,
    ) -> IngestResponse:
        # Extract text with PyMuPDF
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        
        full_text = "\n".join(pages_text)
        
        # Try to parse title from first page
        title = self._extract_pdf_title(pages_text[0] if pages_text else "", filename)
        
        paper_id = f"pdf::{hashlib.md5(full_text[:1000].encode()).hexdigest()[:12]}"
        
        if await paper_exists(paper_id):
            return IngestResponse(
                paper_id=paper_id, title=title, authors=[], year=None,
                entities_found=0, graph_nodes_added=0, already_exists=True
            )

        return await self._run_pipeline(paper_id, title, [], datetime.now().year, full_text, index)

    async def _run_pipeline(
        self, paper_id, title, authors, year, text, index: EmbeddingIndex
    ) -> IngestResponse:
        logger.info("Processing paper: '%s'", title[:50])

        # Step 1: Entity extraction
        extractor = _get_extractor()
        extraction = extractor.extract(text)

        # Step 2: Add to search index
        abstract = text[:1000]
        index.add(paper_id, title, abstract, year, authors)

        # Step 3: Update knowledge graph
        graph = _get_graph()
        nodes_added = graph.add_paper(paper_id, title, authors, year, extraction)

        logger.info(
            "Ingestion complete: %d entities, +%d graph nodes",
            len(extraction.entities), nodes_added
        )
        return IngestResponse(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            entities_found=len(extraction.entities),
            graph_nodes_added=nodes_added,
        )

    def _extract_pdf_title(self, first_page: str, fallback: str) -> str:
        """Heuristic: title is usually the first long line of the first page."""
        lines = [l.strip() for l in first_page.split("\n") if len(l.strip()) > 20]
        return lines[0] if lines else fallback.replace(".pdf", "")
