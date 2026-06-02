"""
/ingest — paper ingestion endpoints (arXiv + PDF upload).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import arxiv
import tempfile, os, uuid

from src.api.schemas import ArxivIngestRequest, IngestResponse
from src.pipeline.ingestion import PaperIngestionPipeline
from src.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)
pipeline = PaperIngestionPipeline()


@router.post("/arxiv", response_model=IngestResponse)
async def ingest_arxiv(body: ArxivIngestRequest, request: Request):
    """
    Fetch a paper from arXiv by ID, run the NLP pipeline,
    and add it to the knowledge graph + search index.
    """
    try:
        search = arxiv.Search(id_list=[body.arxiv_id])
        results = list(search.results())
        if not results:
            raise HTTPException(status_code=404, detail=f"arXiv paper {body.arxiv_id} not found")
        
        arxiv_paper = results[0]
        result = await pipeline.process_arxiv_paper(
            arxiv_paper, 
            index=request.app.state.index,
            include_references=body.include_references
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"arXiv ingestion failed for {body.arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, run NLP pipeline,
    and add to knowledge graph + search index.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    tmp_path = None
    try:
        # Write upload to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = await pipeline.process_pdf(
            tmp_path, 
            filename=file.filename,
            index=request.app.state.index
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/status/{paper_id}")
async def ingest_status(paper_id: str):
    """Check if a paper has already been ingested."""
    from src.pipeline.ingestion import paper_exists
    exists = await paper_exists(paper_id)
    return {"paper_id": paper_id, "ingested": exists}
