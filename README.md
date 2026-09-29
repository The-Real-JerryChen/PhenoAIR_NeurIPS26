# PhenoAIR code 

This repository contains the modular Python implementation of paper "From Retrieval to Reasoning: Agentic Mechanism Prediction from Cell Painting Profiles" (NeurIPS 2026). The same engine supports GPT or Claude and all three released MoA levels.

> **Reproduction status:** the package interface, released-data loader, agent
> orchestration, controller registry, local retrieval cache, and self-evolution
> building blocks are runnable. The default runtime is not yet the exact pipeline
> used for the paper's reported numbers: it currently uses uncalibrated CellCLIP
> cosine retrieval and RDKit structure retrieval, and does not yet consume the
> calibration/S-policy assets or the original external mechanism-data adapters.

## Install

```bash
cd PhenoAIR
python -m pip install -e .
```

Download the released datasets from
[Jerrychen229/PhenoAIR_MOA](https://huggingface.co/datasets/Jerrychen229/PhenoAIR_MOA)
into the code repository root:

```bash
python -m pip install -U huggingface_hub
hf download Jerrychen229/PhenoAIR_MOA --repo-type dataset --local-dir data
```

Commands are expected to run from this repository root. Relative paths are
resolved from the current working directory; the defaults are `data/` for the
datasets and `outputs/` for predictions. A standard checkout therefore has the
following layout:

```text
PhenoAIR/
|-- data/
|   |-- moa_verified/
|   |-- moa_extended/
|   `-- moa_novel/
|-- examples/
|-- phenoair/
|-- outputs/                 # created on first run
|-- pyproject.toml
`-- run_phenoair.py
```

The data may also live elsewhere; pass `--data-dir /path/to/data`. This is
recommended when the code and Hugging Face dataset are downloaded separately.

## Run

For Azure OpenAI (GPT5.1):

```bash
export AZURE_API_KEY=...
export AZURE_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
export AZURE_DEPLOYMENT=YOUR-GPT-DEPLOYMENT

phenoair --level moa_verified --model gpt --max-workers 4
```

For Claude:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-sonnet-4-6

phenoair --level moa_novel --model claude --max-workers 4
```

Available levels are `moa_verified`, `moa_extended`, and `moa_novel`. Useful controls include `--selected-id`, `--max-examples`, `--max-refine-rounds`, `--rules-file`, and `--arbiter-heuristics-file`.

Claude tool submission always uses `tool_choice="auto"`.
The paper's Claude 4 experiments used the now-retired
`claude-sonnet-4-20250514` snapshot. The runnable default follows
[Anthropic's replacement model](https://docs.anthropic.com/en/docs/about-claude/model-deprecations),
`claude-sonnet-4-6`; results from it are a new backbone run, not an exact
reproduction of the reported Claude 4 numbers.

## Local tool cache

Deterministic retrieval results are cached under `.cache/phenoair/` by default.
The cache is local runtime state and is not part of the source or dataset
release. Configure it with:

```bash
phenoair --level moa_novel --model gpt \
  --cache-dir .cache/phenoair \
  --cache-policy read-write
```

Available policies are `off`, `read-only`, `read-write`, and `refresh`. Cache
entries are content-addressed by tool inputs, dataset fingerprint, and tool
version, so incompatible results are not silently reused. See
[CACHE_POLICY.md](CACHE_POLICY.md) for the artifact release policy and
third-party-data exclusions.

## External mechanism data

The paper experiments optionally enriched mechanism evidence with PubChem,
ChEMBL, and DrugBank. These third-party databases are not bundled with PhenoAIR.
Keep manually downloaded resources under `external_data/`, not the released
benchmark `data/` directory. `external_data/` is excluded by `.gitignore`.

| Source | Access | Local placement |
|---|---|---|
| [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) | Queried on demand; no account or bulk download is required | Recommended future cache namespace: `.cache/phenoair/compound_metadata/` |
| [ChEMBL downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads) | Download the SQLite release manually, or use [`chembl-downloader`](https://github.com/cthoyt/chembl-downloader) | `external_data/chembl/chembl_<release>.db` |
| [DrugBank academic access](https://go.drugbank.com/academic_research/) | Requires the user's own DrugBank account and applicable license; download the full XML archive from the official portal | `external_data/drugbank/full_database.xml` |

Recommended layout:

```text
PhenoAIR/
|-- data/                         # released PhenoAIR benchmark data
|-- external_data/                # user-provided; never committed
|   |-- chembl/
|   |   `-- chembl_<release>.db
|   `-- drugbank/
|       |-- full_database.xml
|       `-- parsed/
|           |-- chemistry.csv
|           |-- pharmacology.csv
|           `-- protein_associations.csv
`-- .cache/phenoair/              # generated tool results
```

For the original ChEMBL adapter, point `CHEMBL_SQLITE_PATH` to the downloaded
SQLite file:

```bash
export CHEMBL_SQLITE_PATH="$PWD/external_data/chembl/chembl_<release>.db"
```

Do not upload DrugBank XML, parsed DrugBank tables, or a merged compound cache to
GitHub or Hugging Face. Each user must obtain DrugBank independently under their
own license. PubChem and ChEMBL data should retain source/version provenance and
the required attribution described by their providers.

> **Current implementation status:** the modular release currently uses released
> SMILES plus RDKit structure retrieval. The PubChem/ChEMBL/DrugBank adapters and
> the DrugBank XML parser are being separated from the original notebooks and are
> not yet consumed by the default runtime. Placing files in `external_data/` does
> not change current predictions until those adapters are enabled.

## Modules

| Module | Public responsibility |
|---|---|
| `phenoair.data` | Load released tables and aggregate well-level profiles |
| `phenoair.retrieval` | CellCLIP phenotype retrieval and RDKit structure retrieval |
| `phenoair.reference_memory` | Load and verify released reference reliability memory |
| `phenoair.agents` | Provider-neutral tool loop and P/M/A agents |
| `phenoair.memory` | Candidate-memory state and eligibility |
| `phenoair.controller` | Formal rules, mutable registry, structured-rule parser |
| `phenoair.engine` | Single-case and concurrent dataset orchestration |
| `phenoair.evolution` | Trajectory compression, prompts, proposals, optional application |

The components are constructor arguments to `PhenoAIR`, so callers can replace an agent, retriever, prompt, or registry without editing the engine.

## Reference reliability memory

Sanitized reference-only reliability memory is bundled as a versioned method
asset. It contains the final source, MoA, reference-anchor, and intra-family
confusion judgments exported from the phenotype-calibration workflow. It does not
contain prompts, raw LLM responses, generation traces, test trajectories, or
absolute paths. The current default retriever does not consume this asset yet;
it is exposed for custom calibration implementations and the forthcoming exact
reproduction path.

```python
from phenoair import load_reference_memory

memory = load_reference_memory("moa_verified")
raf = memory.moa("RAF inhibitor")
gnf5 = memory.reference("GNF-5")
```

The extended level has its own memory. The novel level explicitly reuses the
verified reference memory because both use the same known-MoA reference panel.
Each asset includes a reference-panel fingerprint and a content fingerprint;
the loader verifies the content fingerprint before returning any entries.

## Customize rules

```python
from phenoair.controller import build_default_registry, rule_from_spec

registry = build_default_registry()
registry.remove("inspect_distribution_once")
registry.set_enabled("expand_low_reliability", False)

registry.add(rule_from_spec({
    "rule_id": "inspect_supported_medium_top",
    "priority": 150,
    "conditions": [
        {"path": "p.retrieval_reliability", "op": "eq", "value": "medium"},
        {"path": "m.top_verdict", "op": "eq", "value": "support"}
    ],
    "action": {
        "type": "GET_BROADER_SUMMARY",
        "parameters": {"max_k": 50},
        "reason": "inspect broader evidence before stopping"
    }
}, source="user"))
```

Rules can also be loaded from JSON with `registry.load_specs(...)` or the CLI `--rules-file`. Lower numeric priorities fire first. Registry operations include `add`, `remove`, `get`, `set_enabled`, `iter_rules`, and `snapshot`.

## Customize agents

```python
from phenoair import PhenoAIR
from phenoair.controller import build_default_registry

system = PhenoAIR(
    dataset=my_dataset,
    phenotype_agent=my_phenotype_agent,
    mechanism_agent=my_mechanism_agent,
    arbiter_agent=my_arbiter_agent,
    registry=build_default_registry(),
    max_refine_rounds=4,
)
```

`PhenotypeRetriever` is a protocol, so a custom retrieval backend only needs to implement `neighbors`, `distribution`, and `candidate_support`.

## Self-evolution building blocks

The release does not prescribe evolution rounds or ship rules generated by a particular run. `EvolutionEngine` provides independent operations:

```python
analysis = evolver.analyze_failures(failure_records, success_records)
proposal = evolver.propose_controller_rules(cluster, positives, negatives)
proposal.save(Path("proposal.json"))

# Applying generated artifacts is explicit.
proposal.apply_rules(registry, selected_rule_ids={"chosen_rule"})
```

Arbiter evolution returns candidates for only the mutable decision-heuristics section. The role definition and output schema remain fixed. Prior changes and result-change summaries can be supplied as optional context, without imposing a directory or round convention.

See [examples/customize_controller.py](examples/customize_controller.py) and [examples/evolve_rules.py](examples/evolve_rules.py).

## Outputs

Predictions are written to `outputs/<level>/<model>/predictions.jsonl` and `predictions.json`. Each record contains the complete P/M/controller trajectory, final memory, Arbiter output, and tool log. Successful existing JSONL records are reused, error records are retried automatically, and `--force` reruns every requested case.

## Citation

If you find this repo useful, please cite:

```bibtex
@inproceedings{chen2026phenoair,
  title     = {From Retrieval to Reasoning: Agentic Mechanism Prediction from Cell Painting Profiles},
  author    = {Chen, Jiayuan and Yu, Botao and Liu, Tianyu and Pham, Thai-Hoang and Wu, Meng and Zhang, Ping},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
