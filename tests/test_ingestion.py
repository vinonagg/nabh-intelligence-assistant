from src.ingestion import chunk_pages

def test_chunk_pages_preserves_page_numbers():
    pages = [{"page": 7, "text": "NABH standards " * 150}]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    assert chunks
    assert all(chunk.page_start == 7 and chunk.page_end == 7 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
