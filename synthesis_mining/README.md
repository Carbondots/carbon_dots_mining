# Synthesis Mining Pipeline

This workflow is split across two folders:

- `preprocess/`: local steps `01` to `04`, shared with the property workflow
- `synthesis_mining/`: local steps `01` to `09`, corresponding to global pipeline steps `05` to `13`

If you want the shared preprocessing details, read [`../preprocess/README.md`](../preprocess/README.md) first.

## Required Folder Pattern

- Input PDFs: `data/pdfs/*.pdf`
- Working root: `data/mining/<document_id>/...`
- Final output: `data/mining/<document_id>/Synthesis/letter_table/<document_id>.csv`

## Local Step Chain

1. `01_cos_tokenize.py`  
   Pipeline step `05`  
   Input: `preprocess/Tokenized/<id>.csv`  
   Output: `Synthesis/cos_tokenized/<id>.csv|.txt|.md`
2. `02_similarity_filter.py`  
   Pipeline step `06`  
   Input: `Synthesis/cos_tokenized/<id>.csv` and `experiment_templates.txt`  
   Output: `Synthesis/cos/<id>.csv|.txt|.md`
3. `03_llm_synthesis_decision.py`  
   Pipeline step `07`  
   Input: `Synthesis/cos/<id>.csv`  
   Output: `Synthesis/LLM_decision_32b/<id>.csv|.txt|.md`
4. `04_llm_refine_synthesis.py`  
   Pipeline step `08`  
   Input: `Synthesis/LLM_decision_32b/<id>.csv`  
   Output: `Synthesis/LLM_abstract_qwen2.5vl/<id>.md`
5. `05_llm_extract_sample_names.py`  
   Pipeline step `09`  
   Input: `Synthesis/LLM_abstract_qwen2.5vl/<id>.md` and `Synthesis/LLM_decision_32b/<id>.csv`  
   Output: `Synthesis/LLM_name_qwen2.5vl/<id>.md`
6. `06_llm_extract_tables.py`  
   Pipeline step `10`  
   Input: `Synthesis/LLM_name_qwen2.5vl/<id>.md`  
   Output: `Synthesis/LLM_table_qwen2.5vl/<id>_all_extractions.md`
7. `07_merge_extractions_to_csv.py`  
   Pipeline step `11`  
   Input: `Synthesis/LLM_table_qwen2.5vl/<id>_all_extractions.md`  
   Output: `Synthesis/LLM_table_qwen2.5vl/<id>.csv` and `<id>_review_todo.json`
8. `08_normalize_abbreviation_conflicts.py`  
   Pipeline step `12`  
   Input: merged CSV and review JSON from local step `07`  
   Output: updated merged CSV plus `cascade_review.json`
9. `09_llm_fill_review_fields.py`  
   Pipeline step `13`  
   Input: merged CSV, review JSON, and `Synthesis/LLM_decision_32b/<id>.csv`  
   Output: `Synthesis/letter_table/<id>.csv`

## Shared Modules

- `synthesis_unit.py`: shared path handling, document discovery, text cleanup, review-context loading, markdown-table parsing, synthesis-description construction, and LLM JSON cleanup
- `clean_common.py`: compatibility shim for older imports that still expect `clean_common`

If a helper is not prompt-specific and is needed by multiple synthesis stages, add it to `synthesis_unit.py` instead of duplicating it.

## Integrated Run

Run the full preprocessing + synthesis pipeline from the repository root:

```bash
python synthesis_mining/run_synthesis_pipeline.py --pdf-root data/pdfs --mining-root data/mining
```

`run_synthesis_pipeline.py` still uses global pipeline step numbers `01` to `13`, so the late-stage example remains:

```bash
python synthesis_mining/run_synthesis_pipeline.py --mining-root data/mining --start-step 6 --end-step 13
```

## Dependencies

- Root dependency file: [`../requirements-synthesis.txt`](../requirements-synthesis.txt)
- Local dependency snapshot: [`./requirements.txt`](./requirements.txt)
