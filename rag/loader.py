import fitz  # pymupdf

def load_pdf(file) -> list[dict]:
    '''
    Accepts a Streamlit UploadedFile object.
    Returns a list of { 'text': str, 'page': int }
    '''
    doc = fitz.open(stream=file.read(), filetype="pdf")
    
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:  # skip blank pages
            pages.append({
                'text': text,
                'page': i + 1
            })
    
    return pages