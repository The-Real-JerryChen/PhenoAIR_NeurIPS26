# Cache and method-asset release policy

PhenoAIR separates reproducible method assets from disposable local caches. Cache
contents are never required to live in the source repository and `.cache/` is
ignored by Git.

## Public cache interface

Deterministic tool results use the versioned, content-addressed
`JsonCacheStore`. A cache entry records its namespace, producer version, stable
input key, schema version, creation time, and JSON value. Writes are atomic.

The command-line policies are:

| Policy | Read existing entries | Write new entries |
|---|---:|---:|
| `off` | no | no |
| `read-only` | yes | no |
| `read-write` | yes | yes |
| `refresh` | no | yes |

The default directory is `.cache/phenoair/`, relative to the run root. The
current implementation caches deterministic CellCLIP cosine neighbors and RDKit
Morgan-fingerprint neighbors. Prediction JSONL resume behavior is separate from
the tool cache.

## Existing experimental artifacts

The original experiments contain several artifact classes. They should not be
published as one undifferentiated cache archive.

| Artifact | Approximate size | Release decision |
|---|---:|---|
| Calibrated phenotype KNN | 381 MB | Rebuild locally from released features plus public calibration assets |
| Structure KNN | 122 MB | Rebuild locally from released SMILES |
| Query-level S outputs | Per-level JSON collections | Rebuild from public S-policy configuration and calibrated neighbors |
| Final reference reliability memory | 2.4 MB sanitized | Bundled as a versioned method asset; LLM traces are omitted |
| Drug metadata DuckDB | 1019 MB | Do not publish; it co-mingles records and fields from multiple providers |
| SwissTargetPrediction CSV | 23 MB | Do not publish without explicit redistribution clearance |
| LLM responses and MAS trajectories | Variable | Do not publish as cache or method assets |
| Final predictions | Variable | User-generated output only |

The calibrated KNN, reference memory, and S policy are part of the method, not
mere performance caches. Exact paper reproduction requires sanitized versions
of the final calibration configuration and S-policy configuration in addition
to the bundled reference memory. Method assets must contain no absolute paths,
credentials, model traces, timestamps used as identifiers, or test
ground-truth-dependent fields.

## Third-party data

The existing `drug_cache.duckdb` cannot be redistributed as-is. It contains
merged PubChem, ChEMBL, and DrugBank records without field-level provenance.
DrugBank's current terms restrict making its databases available to third
parties. PubChem notes that reuse conditions can depend on the contributing data
source. ChEMBL is distributed under CC BY-SA 3.0, but mixing it into a database
still requires attribution and share-alike compliance.

SwissTargetPrediction labels its licensed materials CC BY 4.0, while its terms
also restrict automated collection of a substantial portion of those materials.
The existing bulk CSV collection should therefore remain excluded unless SIB
confirms redistribution is permitted for this release.

Relevant terms:

- DrugBank: https://trust.drugbank.com/drugbank-trust-center/terms-of-use
- PubChem downloads and source licensing: https://pubchem.ncbi.nlm.nih.gov/docs/downloads
- ChEMBL: https://www.ebi.ac.uk/chembl/
- SwissTargetPrediction: https://www.swisstargetprediction.ch/termsofuse.php

This policy is an engineering release decision, not legal advice.

## Public and planned method assets

The sanitized reference memory now lives in the code release. Calibration and
S-policy assets still need to be added:

```text
phenoair/resources/
|-- calibration/
|   |-- moa_verified.json
|   |-- moa_extended.json
|   `-- moa_novel.json
|-- reference_memory/
|   |-- moa_verified/memory.json
|   |-- moa_extended/memory.json
|   `-- moa_novel/alias.json
`-- s_policy/
    |-- moa_verified.json
    |-- moa_extended.json
    `-- moa_novel.json
```

`moa_novel` explicitly references the verified reference memory because the
benchmark uses the same known-MoA reference panel. Every asset needs a schema
version, producer version, and input-data fingerprint. Per-query KNN and S-output
files remain local generated cache entries.
