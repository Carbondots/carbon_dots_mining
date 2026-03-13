# Property Mining Pipeline

This folder is a standalone property-extraction pipeline. It does not import the legacy repository and can run directly as long as the mining root already contains per-paper folders with the expected preprocessing outputs.

## Required folder pattern

Each paper folder is expected to live under one mining root, for example:

- `data/mining/<paper_id>_<paper_name>/preprocess/cut_property/<paper_id>.md`, or
- `data/mining/<paper_id>_<paper_name>/preprocess/cut/<paper_id>_cut.md`

The property pipeline writes all stage outputs back into the same paper folder under `property/<stage_name>/`.

## Step chain

1. `01_tokenize_property_text.py`
   Input: `preprocess/cut_property/<id>.md` or `preprocess/cut/<id>_cut.md`
   Output: `property/Tokenized/<id>.csv|.txt|.md`
2. `02_label_property_text.py`
   Input: `property/Tokenized/<id>.csv`
   Output: `property/label_LLM/<id>.csv|.txt|.md`
3. `03_ocr_tables_and_relabel.py`
   Input: `property/label_LLM/<id>.csv`
   Output: `property/label_LLM_vl/<id>.csv|.txt|.md`
4. `04_llm_decide_samples_and_props.py`
   Input: `property/label_LLM_vl/<id>.csv` and `Synthesis/letter_table/<id>.csv`
   Output: `property/decision_LLM/<id>.csv|.txt|.md`
5. `05_llm_refine_property_text.py`
   Input: `property/decision_LLM/<id>.csv`
   Output: `property/abstract_refine/<id>.md` plus `main|vs|app` variants
6. `06_llm_clean_refined_properties.py`
   Input: `property/abstract_refine/*.md`
   Output: `property/abstract_clean/*.md`
7. `07_llm_resolve_multisample_main.py`
   Input: `property/abstract_clean/<id>.md`
   Output: `property/abstract_clean/<id>.md` with resolved main attributions
8. `08_llm_route_main_to_app_vs.py`
   Input: `property/abstract_clean/<id>.md`
   Output: `property/abstract_clean/<id>_main.md`, `property/abstract_clean/<id>_vs.md`, `property/abstract_clean/<id>_app.md`
9. `09_llm_bind_app_vs_samples.py`
   Input: routed `app` and `vs` markdown
   Output: bound `app` and `vs` markdown in the same stage folder
10. `10_llm_review_final_properties.py`
    Input: cleaned `main|app|vs` markdown
    Output: `property/final_reviewed_properties/<id>.md`
11. `11_llm_structure_and_deduplicate_properties.py`
    Input: `property/final_reviewed_properties/<id>.md`
    Output: `property/final_structured_properties/<id>.md`
12. `12_llm_resolve_property_conflicts.py`
    Input: `property/final_structured_properties/<id>.md`
    Output: `property/conflict_resolved_properties/<id>.md`
13. `13_llm_resolve_change_entries.py`
    Input: `property/conflict_resolved_properties/<id>.md`
    Output: `property/change_resolved_properties/<id>.md`
14. `14_export_property_letter_table.py`
    Input: `property/change_resolved_properties/<id>.md`
    Output: `property/letter_table/<id>.csv`

## Shared modules

- `property_unit.py`: shared path handling, markdown parsing/rendering, CSV readers, evidence helpers, sample-name normalization, and LLM output cleanup.
- `pipeline_utils.py`: lightweight shared filters for numeric-property handling.

If multiple steps need the same non-prompt helper, add it to one of these modules instead of duplicating it in a stage script.

## Integrated run

Run the full pipeline:

```bash
python run_property_pipeline.py --root data/mining
```

Run only a subrange:

```bash
python run_property_pipeline.py --root data/mining --start-step 5 --end-step 14
```

## Main dependencies

- `pandas`
- `tqdm`
- `matplotlib`
- `spacy`
- `lmstudio`
