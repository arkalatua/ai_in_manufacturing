# CMP Digital Twin (ME6324)

A Streamlit "fab simulator". It replays 791 real CMP wafers in time order. These wafers come from day 40.7 to day 65 of the PHM 2016 data, and the twin never saw them during training. For each wafer the twin:
1. predicts MRR with an uncertainty band,
2. splits the prediction into Preston physics, a bounded AI residual and a metrology bias correction,
3. decides whether the wafer needs physical metrology,
4. learns from every wafer it did measure.

## Folder contents

| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `twin_core.py` | Active metrology loop, Bayesian bias assimilation, and a numpy forward pass of the 5-model ensemble |
| `generate_demo_data.py` | Rebuilds `demo_data/` from the raw dataset (about 3–4 min on CPU, needs torch) |
| `demo_data/` | `stream.csv` (per-wafer predictions), `history.csv`, `twin_weights.npz`, `meta.json` |

`generate_demo_data.py` reuses the model code from the project notebook unchanged, with one exception. **All consumable usage counters**
(`u_dresser, u_table, u_film, u_membrane, u_sheet, u_dtable, day`) are masked out of the residual net, so they can only act through the
pad-state ODE. The notebook masked only `day, u_dtable`. With that mask the twin degrades on future wafers (RMSE 6.7). With the full mask it
reproduces the deployment numbers in `results/summary.json`:

| Deployment stream (791 unseen future wafers) | This demo | summary.json |
|---|---|---|
| Hybrid twin, no assimilation | 5.33 | 5.27 |
| Preston + pad-state ODE | 5.31 | 5.31 |
| Black-box GBM | 7.82 | 7.83 |
| Hybrid twin + assimilation (all wafers measured) | 4.18 | 4.20 |
| Black-box GBM + assimilation | 6.59 | 6.59 |

To regenerate: `python generate_demo_data.py "/path/to/2016 PHM DATA CHALLENGE CMP DATA SET/"`. At the end it checks that the numpy
model matches the torch model (max difference about 5e-5 nm/min).
