# Dataset preparation

The repository does not redistribute third-party datasets.

## Primary dataset

Download the Corn or Maize Leaf Disease Dataset from:

https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

Prepare the folders as:

```text
data/primary/
├── Common_Rust/
├── Gray_Leaf_Spot/
├── Healthy/
└── Northern_Leaf_Blight/
```

The manuscript reports the following class totals:

| Class | Images |
|---|---:|
| Common Rust | 1,306 |
| Northern Leaf Blight | 1,146 |
| Healthy | 1,162 |
| Gray Leaf Spot | 574 |
| Total | 4,188 |

The manuscript split is 70/15/15 stratified with seed 42.

Run:

```bash
python -m src.data --config configs/experiment.json --make-splits
```

The script will write:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

Each manifest contains an absolute file path, class name, and class index.

## PlantDoc

PlantDoc is used only for cross-dataset evaluation in the manuscript. The exact maize-subset file manifest used by the authors should be committed before public release so another researcher can reproduce the same secondary evaluation without ambiguity.

Do not create or claim a secondary split that you did not actually use.
