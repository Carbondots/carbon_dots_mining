# Carbon Dot Literature Mining Pipelines

This repository organizes the carbon-dot literature workflow into three coordinated folders:

- `preprocess/`: shared preprocessing steps `01` to `04`
- `synthesis_mining/`: synthesis-only steps `01` to `09` corresponding to global pipeline steps `05` to `13`
- `property_mining/`: property-only steps `01` to `14`

The goal is to turn paper-level source files into reusable mining folders, then export structured synthesis and property tables with traceable intermediate results.

## End-to-End Flow

1. `preprocess/` converts source PDFs into document-level folders and shared sentence-level artifacts under `preprocess/`.
2. `synthesis_mining/` consumes the shared preprocessing outputs and writes sample-level synthesis results to `Synthesis/letter_table/<id>.csv`.
3. `property_mining/` reuses the preprocessing outputs together with synthesis-side sample context and writes normalized property outputs to `property/letter_table/<id>.csv`.

## Folder Guides

- [Preprocess README](./preprocess/README.md)
- [Synthesis Mining README](./synthesis_mining/README.md)
- [Property Mining README](./property_mining/README.md)

## Dependency Files

- Synthesis workflow: [`requirements-synthesis.txt`](./requirements-synthesis.txt)
- Property workflow: [`requirements-property.txt`](./requirements-property.txt)
