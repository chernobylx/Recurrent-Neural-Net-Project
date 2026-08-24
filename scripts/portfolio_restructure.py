"""One-shot restructure: move/clean the notebook and extract report figures.

Run from the repository root. Safe to re-run only before the original
notebook has been removed.
"""
import base64
import json
from pathlib import Path

SRC = Path("rnn-project (1).ipynb")
DST = Path("notebooks/disaster-tweet-classification-rnn.ipynb")
FIG_DIR = Path("reports/figures")

nb = json.loads(SRC.read_text())
cells = nb["cells"]


def settext(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)


# --- typo fixes in markdown ---
settext(5, "".join(cells[5]["source"]).replace("Overwiew", "Overview").replace("Kagle", "Kaggle"))
settext(27, "".join(cells[27]["source"]).replace("Preprocessning", "Preprocessing"))

# --- fill empty architecture description sections ---
settext(33, """#### Explanation of Architecture

**Architecture 1 — Simple GRU.** A sequential model: trainable `Embedding` → single unidirectional `GRU` → `Dropout` → `Dense` (ReLU) → sigmoid output. Five tunable hyperparameters: embedding dimension, GRU units, learning rate, dropout rate, and dense units.

**Architecture 2 — Bidirectional GRU.** A functional-API model: trainable `Embedding` → `SpatialDropout1D` → two stacked `Bidirectional(GRU)` layers (each with input dropout, recurrent dropout, and L2 kernel regularization) separated by `BatchNormalization` → concatenated `GlobalMaxPooling1D` + `GlobalAveragePooling1D` → L2-regularized `Dense` (ReLU) with dropout and batch normalization → sigmoid output. Ten tunable hyperparameters.

Both models are compiled with the Adam optimizer and binary cross-entropy loss, tracking accuracy, precision, and recall.
""")
settext(34, """#### Architecture Comparison

| | Architecture 1 | Architecture 2 |
|---|---|---|
| Recurrent layers | 1 × unidirectional GRU | 2 × bidirectional GRU (stacked) |
| Sequence summarization | Final GRU hidden state | Concatenated global max + average pooling |
| Regularization | Dropout only | Spatial dropout, GRU/recurrent dropout, L2, batch normalization |
| Tunable hyperparameters | 5 | 10 |
| Search trials (Bayesian) | 512 | 64 |
""")

# --- data-loading cell: fall back to local data/ when not on Kaggle ---
settext(8, "".join(cells[8]["source"]).replace(
    "data = Path('/kaggle/input/nlp-getting-started')",
    "data = Path('/kaggle/input/nlp-getting-started')\n"
    "if not data.exists():  # running locally: expects train.csv/test.csv in ../data\n"
    "    data = Path('../data')",
))

# --- extract embedded figures ---
FIG_DIR.mkdir(parents=True, exist_ok=True)
figures = {
    (13, 1): "target_distribution",
    (16, 1): "text_feature_analysis",
    (16, 3): "wordclouds",
    (20, 0): "keyword_analysis",
    (24, 1): "location_analysis",
    (51, 1): "model_comparison",
}
for (ci, oi), name in figures.items():
    png = cells[ci]["outputs"][oi]["data"]["image/png"]
    (FIG_DIR / f"{name}.png").write_bytes(base64.b64decode(png))

# --- drop trailing empty code cell ---
if not "".join(cells[-1]["source"]).strip():
    cells.pop()

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, indent=1))
SRC.unlink()
print(f"Wrote {DST}, extracted {len(figures)} figures, removed {SRC.name}")
