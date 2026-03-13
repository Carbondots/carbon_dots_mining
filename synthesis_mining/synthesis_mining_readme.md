# Synthesis Mining Pipeline

This folder is a standalone synthesis-parameter extraction pipeline. It does not import the legacy repository. All numbered steps only depend on files created earlier in the same folder plus third-party packages.

## Required folder pattern

The pipeline starts from source PDFs and writes one document folder per paper:

- Input PDFs: `data/pdfs/*.pdf`
- Working root: `data/mining/<document_id>/...`
- Final output: `data/mining/<document_id>/Synthesis/letter_table/<document_id>.csv`

## Step chain

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
   Output: `Synthesis/Tokenized/<id>.csv|.txt|.md`
5. `05_similarity_filter.py`
   Input: `Synthesis/Tokenized/<id>.md` and `experiment_templates.txt`
   Output: `Synthesis/cos/<id>.csv|.txt|.md`
6. `06_llm_synthesis_decision.py`
   Input: `Synthesis/cos/<id>.csv`
   Output: `Synthesis/LLM_decision_32b/<id>.csv|.txt|.md`
7. `07_llm_refine_synthesis.py`
   Input: `Synthesis/LLM_decision_32b/<id>.csv`
   Output: `Synthesis/LLM_abstract_qwen2.5vl/<id>.md`
8. `08_llm_extract_sample_names.py`
   Input: `Synthesis/LLM_abstract_qwen2.5vl/<id>.md` and `Synthesis/LLM_decision_32b/<id>.csv`
   Output: `Synthesis/LLM_name_qwen2.5vl/<id>.md`
9. `09_llm_extract_tables.py`
   Input: `Synthesis/LLM_name_qwen2.5vl/<id>.md`
   Output: `Synthesis/LLM_table_qwen2.5vl/<id>_all_extractions.md`
10. `10_merge_extractions_to_csv.py`
    Input: `Synthesis/LLM_table_qwen2.5vl/<id>_all_extractions.md`
    Output: `Synthesis/LLM_table_qwen2.5vl/<id>.csv` and `<id>_review_todo.json`
11. `11_normalize_abbreviation_conflicts.py`
    Input: merged CSV and review JSON from Step 10
    Output: updated Step 10 CSV and `cascade_review.json`
12. `12_llm_fill_review_fields.py`
    Input: Step 10 CSV, Step 10 review JSON, and `Synthesis/LLM_decision_32b/<id>.csv`
    Output: `Synthesis/letter_table/<id>.csv`

## Shared modules

- `synthesis_unit.py`: shared path display, document discovery, text cleanup, review-context loading, markdown-table parsing, synthesis-description construction, and LLM JSON cleanup.
- `clean_common.py`: compatibility shim for older imports that still expect `clean_common`.

If a helper is not prompt-specific and is needed by multiple steps, add it to `synthesis_unit.py` instead of copying it into another step.

## Integrated run

Run the full pipeline:

```bash
python run_synthesis_pipeline.py --pdf-root data/pdfs --mining-root data/mining
```

Run only later stages:

```bash
python run_synthesis_pipeline.py --mining-root data/mining --start-step 5 --end-step 12
```

## Main dependencies

- `numpy`
- `pandas`
- `scikit-learn`
- `spacy`
- `sentence-transformers`
- `torch`
- `pylatexenc`
- `lmstudio`
- `magic-pdf`
