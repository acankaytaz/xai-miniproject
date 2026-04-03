# Validating GNN Explainability via Logical Rule Recovery

A framework for providing **global explanations** for Graph Neural Networks (GNNs) by recovering human-readable logical rules using Description Logics. Built on top of the [EDGE](https://github.com/dice-group/EDGE) evaluation framework with custom adaptations for a video game knowledge graph.

> Group mini-project for **M.079.4091 -- Explainable Artificial Intelligence** (Masters, Summer Semester 2025), Paderborn University. Supervised by Prof. Dr. Stefan Heindorf.

## Overview

GNNs are powerful but operate as black boxes. This project bridges neural prediction with symbolic reasoning: we train a GNN on a knowledge graph, then use concept learners to recover interpretable logical rules that explain the GNN's classification decisions.

**The core experiment:** We define a binary classification task -- predicting whether a video game is "multi-genre" (has 2+ genres) -- using a rule-based ground truth. We then test whether symbolic explainers can recover the original rule (`>= 2 hasGenre.T`) from the GNN's predictions alone.

### Pipeline

```
Wikidata (SPARQL) --> CSV --> RDF/OWL Knowledge Graph --> DGL Heterogeneous Graph --> GNN Training --> Explainer --> Logical Concept
```

1. **Data collection**: SPARQL query harvests ~20,000 rows of video game metadata from Wikidata (genres, developers, publishers, platforms, countries, release dates)
2. **KG construction**: CSV is converted to RDF triples using `rdflib`, producing a knowledge graph with 1,657 nodes and 4,825 edges across 6 node types
3. **GNN training**: R-GCN or R-GAT is trained on a binary node classification task (multi-genre vs. single-genre) with stratified 50/25/25 splits
4. **Explanation**: Logical explainers (EvoLearner, CELOE) and subgraph explainers (PGExplainer, SubGraphX) generate explanations from the GNN's predictions
5. **Evaluation**: Explanations are evaluated on prediction quality (vs. ground truth) and fidelity (vs. GNN predictions)

### Key Result

EvoLearner consistently recovers the ground-truth rule `>= 2 hasGenre.T` ("a game with at least two genres") across all runs, achieving **perfect prediction F1 (1.00)** against ground truth and **0.58 fidelity F1** against the GNN -- confirming that the symbolic explainer captures the intended semantics.

## Knowledge Graph Schema

| Node Type   | Count | Description                  |
|-------------|-------|------------------------------|
| Game        | 299   | Video game entities          |
| Genre       | 107   | Game genres (FPS, RPG, etc.) |
| Developer   | 154   | Development studios          |
| Publisher   | 153   | Publishing companies         |
| Platform    | 127   | Gaming platforms             |
| Country     | 20    | Countries of origin          |

**Edge types**: `hasGenre`, `developedBy`, `publishedBy`, `availableOn`, `hasCountry`, `hasReleaseDate`

## Dataset
**Videogame**
A custom RDF-based dataset containing metadata about various video games, genres, developers, and platforms. This dataset is used for benchmarking explainability techniques on graph-based models.

---

## Installation Guide for the EDGE Framework

Follow these steps to set up the EDGE environment on your system:

### Step 1: Clone the Repository
```bash
git clone https://github.com/21Siddy/xai-miniproject.git
cd xai-miniproject
```

### Step 2: Install Conda
If you don't have Conda installed, download and install it from Anaconda's official website.

### Step 3: Create and Activate the Conda Environment
```bash
conda create --name edge python=3.10
conda activate edge
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Install DGL
```bash
conda install -c dglteam/label/th23_cu121 dgl
```

### Step 6: Test with a command similar to this
```bash
python main.py --datasets videogames --explainers EvoLearner --model RGCN --train --num_runs 5 --print_results
```

---

## Usage

### Train and evaluate with a single explainer

```bash
python main.py --datasets videogames --explainers EvoLearner --model RGCN --train --num_runs 5 --print_results
```

### Run all explainers on multiple datasets

```bash
python main.py --datasets mutag aifb bgs --explainers EvoLearner CELOE PGExplainer SubGraphX --model RGCN --train --num_runs 5 --print_results
```

### CLI Arguments

| Argument          | Default                          | Description                                             |
|-------------------|----------------------------------|---------------------------------------------------------|
| `--datasets`      | `mutag aifb bgs`                | Datasets to evaluate (`mutag`, `aifb`, `bgs`, `videogames`) |
| `--explainers`    | `EvoLearner SubGraphX PGExplainer CELOE` | Explainers to run                            |
| `--model`         | `RGCN`                           | GNN model (`RGCN` or `RGAT`)                           |
| `--train`         | `false`                          | Run training + explanation pipeline                    |
| `--num_runs`      | `5`                              | Number of independent runs                             |
| `--print_results` | `false`                          | Print aggregated result tables                         |

### Sample Results
```
Datasets: ['videogames']
Explainers ['EvoLearner']
Model name: RGCN
Running explainers for 5 runs for dataset videogames
Starting the  Run 1
Starting VideoGameDataset.process() (Full Custom Implementation).
Parsing data/videogames/videogame_f.rdf
Parsing data/KGs/videogame_f.rdf
Prepared 4825 raw rdflib triples after pre-scan.
Generated 117 MultiGenre labels and 182 SingleGenre labels.
VideoGameDataset.process() completed. Final graph has 1657 nodes.
Train/Valid/Test split sizes: Train=149, Valid=74, Test=76
Initializing RGCN  model
Start training...
Epoch 00000 | Train Acc: 0.6107 | Train Loss: 0.6759 | Valid Acc: 0.6081 | Valid loss: 0.6763
...
Early stopping
End Training
Final validation accuracy of the model RGCN on unseen dataset: 0.631578947368421
Training EvoLearner (explanation) on videogames
≥ 2 hasGenre.Genre      Quality:0.84753 Length:4   |Indv.|:117
Total time taken for EvoLearner (explanation)  on videogames: 10.45
```

## Results Summary (Phase 2 -- 299 Games)

### EvoLearner on Videogames (RGCN, mean over 5 runs)

| Metric    | Prediction (vs. Truth) | Fidelity (vs. GNN) |
|-----------|----------------------|---------------------|
| Accuracy  | 1.00                 | 0.63 (+/- 0.02)    |
| Precision | 1.00                 | 0.63 (+/- 0.02)    |
| Recall    | 1.00                 | 0.53 (+/- 0.03)    |
| F1        | 1.00                 | 0.58 (+/- 0.02)    |

**Recovered logical explanation**: `>= 2 hasGenre.T` (stable across all runs)

### Phase 1 vs. Phase 2

| Phase | Games | GNN Test Acc. | Overfitting? | Concept Stability |
|-------|-------|---------------|--------------|-------------------|
| 1     | 33    | 0.56 -- 0.89  | Severe       | Core rule stable, noisy auxiliary conditions |
| 2     | 299   | 0.62 -- 0.72  | Reduced      | Clean `>= 2 hasGenre.T` every run           |

Scaling from 33 to 299 games mitigated GNN overfitting and yielded cleaner, more stable explanations.

## Project Structure

```
.
|-- main.py                     # CLI entry point
|-- src/
|   |-- Explainer.py            # Core orchestrator: trains GNN, runs explainers, evaluates
|   |-- explainer_runner.py     # Multi-run experiment loop
|   |-- gnn_model/
|   |   |-- RGCN.py             # Relational Graph Convolutional Network
|   |   |-- GAT.py              # Relational Graph Attention Network (R-GAT)
|   |   |-- configs.py          # Hyperparameter configurations per dataset/model
|   |   |-- dataset.py          # Dataset loader (includes custom VideoGameDataset)
|   |   |-- hetero_features.py  # Heterogeneous node feature initialization
|   |   `-- utils.py            # Metrics, learning problem construction
|   |-- logical_explainers/
|   |   |-- EvoLearner.py       # EvoLearner concept learner wrapper
|   |   `-- CELOE.py            # CELOE concept learner wrapper
|   `-- dglnn_local/
|       |-- RDFDataset.py       # Custom RDF-to-DGL graph parsing
|       `-- subgraphx.py        # SubGraphX explainer for heterogeneous graphs
|-- configs/
|   `-- pgexplainer.yaml        # PGExplainer hyperparameters
|-- data/
|   |-- KGs/                    # Knowledge graph files (.owl, .rdf)
|   `-- videogames/             # Custom videogame dataset (RDF files + DGL cache)
|-- results/
|   |-- evaluations/            # JSON evaluation metrics per model/explainer/dataset
|   |-- predictions/            # CSV/JSON predictions per run
|   `-- exp_visualizations/     # Explanation visualizations (subgraph plots, CE images)
|-- preprocess_kg.py            # CSV-to-RDF conversion script
|-- postprocess_kg.py           # Post-processing utilities for KG
|-- preprocess.sh               # Shell script for data preprocessing steps
|-- exp_visualize.py            # Explanation visualization script
`-- tests/                      # Unit tests for dataset conversions
```

## Supported Models and Explainers

### GNN Models
- **R-GCN** (Relational Graph Convolutional Network) -- learns relation-specific weight matrices via basis decomposition
- **R-GAT** (Relational Graph Attention Network) -- extends GATv2 to heterogeneous graphs with per-relation attention

### Explainers
| Explainer      | Type       | Output                          |
|----------------|------------|---------------------------------|
| **EvoLearner** | Logical    | Description Logic class expression (global) |
| **CELOE**      | Logical    | Description Logic class expression (global) |
| **PGExplainer** | Subgraph  | Edge importance masks (local)   |
| **SubGraphX**  | Subgraph   | Important subgraph via Shapley values (local) |

## Built With

- [DGL](https://www.dgl.ai/) -- Deep Graph Library for heterogeneous graph neural networks
- [Ontolearn](https://github.com/dice-group/Ontolearn) -- Concept learning framework (EvoLearner, CELOE)
- [RDFLib](https://rdflib.readthedocs.io/) -- RDF graph parsing and serialization
- [PyTorch](https://pytorch.org/) -- Neural network backend

## Report

The full project report is available as [XAi_Miniproject-Report.pdf](./XAi_Miniproject-Report.pdf).

## Authors

- **Siddharth Hemnani** -- Universitat Paderborn
- **Vedant Vallakati** -- Universitat Paderborn
- **Ahmet Can Kaytaz** -- Universitat Paderborn

## References

1. Sapkota, R., Kohler, D., & Heindorf, S. (2024). *EDGE: Evaluation framework for logical vs. subgraph explanations for node classifiers on knowledge graphs.* CIKM, ACM.
2. Schlichtkrull, M., et al. (2018). *Modeling relational data with graph convolutional networks.* ESWC.
3. Koplenig, E., et al. (2023). *EvoLearner: Evolutionary search for global description logic explanations.* ICTAI.

## License

This project was developed for academic purposes as part of a university course.
