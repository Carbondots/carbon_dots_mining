# Preprocess Pipeline

This folder contains the shared preprocessing stages for the repository. Its local numbering already matches the global pipeline order: `01` to `04`.

## Role in the Repository

- Input source PDFs from `data/pdfs/*.pdf`
- Build document-level working folders under `data/mining/<document_id>/...`
- Produce reusable preprocessing outputs for downstream synthesis and property extraction

## Shared Outputs

- `preprocess/PDF2md/*`
- `preprocess/latex/<id>.md`
- `preprocess/cut/<id>_cut.md`
- `preprocess/Tokenized/<id>.csv|.txt|.md`

`synthesis_mining/` consumes `preprocess/Tokenized/<id>.csv` directly.  
`property_mining/` consumes `preprocess/cut/` or `preprocess/cut_property/` inputs and writes its own property-side outputs back into the same paper folder.

## Step Chain

1. `01_pdf_to_markdown.py`
   Input: `data/pdfs/*.pdf`
   Output: `data/mining/<id>/preprocess/PDF2md/*`
2. `02_latex_to_text.py`
   Input: `preprocess/PDF2md/<id>.md`
   Output: `preprocess/latex/<id>.md`
3. `03_trim_markdown_tail.py`
   Input: `preprocess/latex/<id>.md`
   Output: `preprocess/cut/<id>_cut.md`
4. `04_tokenize_markdown.py`
   Input: `preprocess/cut/<id>_cut.md`
   Output: `preprocess/Tokenized/<id>.csv|.txt|.md`

## Typical Run Order

Run from the repository root:

```bash
python preprocess/01_pdf_to_markdown.py --pdf-root data/pdfs --mining-root data/mining
python preprocess/02_latex_to_text.py --mining-root data/mining
python preprocess/03_trim_markdown_tail.py --mining-root data/mining
python preprocess/04_tokenize_markdown.py --mining-root data/mining
```

## Downstream Guides

- [Synthesis Mining README](../synthesis_mining/README.md)
- [Property Mining README](../property_mining/README.md)
