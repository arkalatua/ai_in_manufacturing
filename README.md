# CMP Digital Twin: live demo (ME6324, Group 37)

A Streamlit "fab simulator". It replays 791 real CMP wafers in time order. These wafers come from day 40.7 to day 65 of the PHM 2016 data, and the twin never saw them during training. For each wafer the twin:
1. predicts MRR with an uncertainty band,
2. splits the prediction into Preston physics, a bounded AI residual and a metrology bias correction,
3. decides whether the wafer needs physical metrology,
4. learns from every wafer it did measure.

## Run it (offline, no GPU, no torch)

```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501. Everything it needs is already in `demo_data/` (about 0.8 MB).
`demo_backup.mp4` is a 100-second recording of the full demo. Keep it open in another tab in case anything fails during the presentation.

## Folder contents

| File | What it is |
|---|---|
| `app.py` | The Streamlit app (5 tabs) |
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

## Headline numbers (default policy: measure if σ > 4.4 nm/min)

* The twin skips metrology on **42%** of wafers.
* On those skipped wafers: **twin RMSE 3.97 nm/min, black-box GBM 6.43** (same wafers, same metrology feedback).
* Worst error on a skipped wafer: **11.6 nm/min for the twin**, against 14.6 if the same number of wafers is skipped at random, and 23.7 for GBM.
* At σ > 5.0, **62%** of wafers skip with RMSE 4.42.

## Suggested 4-minute script

1. **Set the scene (20 s).** "You're the process engineer. Wafers come off the CMP tool every few minutes. MRR is never measured in-line, so metrology checks only some of them. The twin was trained on the first 40 days, and we now replay the next 25 days it has never seen."
2. **Play (60 s).** Point out the black dots (true MRR on skipped wafers) sitting inside the blue band, and the red × marks (wafers the twin chose to measure). Then the wafer card on the right: the decision badge and the bar chart showing *Preston physics + AI residual + metrology bias = prediction*.
3. **Turn on "Overlay black-box GBM" and keep playing (40 s).** The first ~5 deployment days (day 40.7–46) run on a **worn dresser near end of life**. In this region the black-box is off by about 8 nm/min and consistently under-predicts. The twin stays at about 4.4 because its pad-state ODE knows about wear. Then play across the **dresser change** (orange line, around day 46): the twin's k_p resets with it. This is your "robust extrapolation under worn-pad conditions" outcome, shown live.
4. **Physics vs AI tab (30 s).** "96% of every prediction is physics. The neural net adds about 3 nm/min and is hard-bounded at ±15%."
5. **Pad state tab (20 s).** The latent k_p sawtooth with the dresser reset, and the recovered wear law checked against unseen wafers.
6. **Metrology savings tab: drag the σ slider (45 s).** Skip goes from 42% to 62%, and the blue curve (uncertainty-guided) stays below random sampling. Then the cost calculator, for example "at 100k wafers per month this is about $6M a year": your $5–20M opex slide.
7. **What-if tab (20 s).** Move the dresser-usage slider and watch MRR decay. "Recipe tuning without burning test wafers."
8. **Close (15 s).** "Greenfield fabs like Micron Sanand and Tata-PSMC Dholera need this tooling built from scratch."

## Presenting tips
* Start the app before class, then press **Reset**.
* At 1280×720 projector resolution, zoom the browser to 80–90% (Ctrl/Cmd −). You can collapse the sidebar after pressing Play.
* Use the **Tool mode shown** radio to switch between A123, A456 and B456. B456 has the strongest drift, so it makes the best story.
* The cost figures in the savings tab are illustrative assumptions you can edit live. Present them as such.
