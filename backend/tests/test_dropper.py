# tests/test_dropper.py
import time
from dropper import get_pipeline

def test_docling_integration():
    sample_pdf = "data/uploads/2408.09869v4.pdf"
    
    """End-to-end test of DoclingExtractor via public API get_pipeline."""
    # Build a pipeline that uses Docling
    pipeline = get_pipeline('pdf', {
        'pdf_pipeline_opts': {'use_docling': True}
    })

    # Prepare input
    data = {'source': sample_pdf}

    # Time the extraction
    start = time.time()
    result = pipeline.execute(data)
    duration = time.time() - start
    print(f"Docling extraction took {duration:.1f}s")

    # Validate output
    doc = result.get('doc')
    assert isinstance(doc, str)
    assert "## Docling Technical Report" in doc

def test_gemini_integration():
    sample_pdf = "data/uploads/2408.09869v4.pdf"
    pipeline = get_pipeline('pdf', {
        'pdf_pipeline_opts': {'use_gemini': True}
    })
    
    data = {'source': sample_pdf}
    start = time.time()
    result = pipeline.execute(data)
    duration = time.time() - start
    print(f"Gemini extraction took {duration:.1f}s")
    
    doc = result.get('doc')
    assert isinstance(doc, str)
    assert "## Docling Technical Report" in doc
    
    
    
    