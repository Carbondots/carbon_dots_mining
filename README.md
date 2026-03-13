# Carbon Dot Literature Mining Pipelines

This repository provides two coordinated text-mining pipelines for carbon dot literature. The project is designed to transform paper-level source documents into structured outputs for synthesis conditions and reported properties.

At a high level, the repository supports the following tasks:

- preprocessing article text into reusable document-level mining folders,
- extracting sample-level synthesis parameters into structured tables,
- extracting sample-level property statements and converting them into normalized letter-table outputs.

## Repository Scope

The repository is organized around two complementary workflows:

- **Synthesis mining** converts article PDFs into structured synthesis records, including sample names, reaction conditions, reagents, solvents, and purification information.
- **Property mining** processes pre-cut article text together with synthesis-side sample information to identify, refine, resolve, and export reported property statements.

Both workflows operate on per-paper directory structures and write stepwise outputs back into the corresponding mining folders, which makes the full process traceable from raw document inputs to final tabular results.

## Workflow Overview

The overall project flow can be summarized as:

1. Article text is prepared and organized into document-specific mining folders.
2. The synthesis pipeline identifies synthesis-related passages and exports structured synthesis tables for each paper.
3. The property pipeline uses document evidence and sample context to extract and normalize reported property information.
4. Final outputs are written as paper-level structured CSV files inside the corresponding workflow folders.

## Pipeline Guides

- [Synthesis Mining README](./synthesis_mining/synthesis_mining_readme.md)
- [Property Mining README](./property_mining/property_mining_readme.md)
