# CMP Digital Twin

**A physics-informed digital twin for chemical-mechanical polishing (CMP), with uncertainty-guided metrology.**

The twin predicts each wafer's material removal rate (MRR) and its own uncertainty. It then decides whether the
wafer needs a physical measurement. This repo includes a live Streamlit dashboard that replays **791 real wafers
the model has never seen**, in time order, the way they would come off a fab tool.

| | Twin | Black-box GBM |
|---|---|---|
| Wafers that skip metrology (σ limit 4.4 nm/min) | **42%** | – |
| RMSE on those skipped wafers | **3.97 nm/min** | 6.43 nm/min |
| Worst error on a skipped wafer | **11.6 nm/min** | 23.7 nm/min |

Both models receive the same metrology feedback, and errors are measured on the same wafers. If the same share of
wafers were skipped at random, the worst error would be 14.6 nm/min.

---

## Quick start

No GPU or PyTorch is needed. Everything the app uses is already in `demo_data/` (about 0.8 MB).

```bash
git clone https://github.com/arkalatua/ai_in_manufacturing.git
cd ai_in_manufacturing
pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at <http://localhost:8501>. Press **Play** in the sidebar to start the wafer stream.
`demo_backup.mp4` is a 100-second screen recording of a full replay.

---

## How the twin works

Each prediction has three parts:

```
MRR  =  Preston physics  ×  pad state  +  AI residual (bounded)  +  metrology bias (assimilated)
```

1. **Preston backbone.** MRR = k<sub>p</sub> · P · V, where P is pressure and V is platen speed. This gives the
   physically expected removal rate.
2. **Latent pad-state ODE.** k<sub>p</sub> decays as the dresser disk dulls and the pad glazes, and it resets when
   the dresser is changed. All consumable-usage counters act **only** through this ODE, so the twin extrapolates
   sensibly to worn pads it has not seen.
3. **Bounded residual network.** A 5-member deep ensemble corrects what the physics misses. Its output is
   hard-limited to ±15% of the mode mean. On average about 96% of each prediction comes from physics.
4. **Bayesian bias assimilation.** Every measured wafer updates a running bias estimate for its tool mode.
5. **Active metrology.** Total uncertainty σ (epistemic + aleatoric + bias) is compared with a policy limit. Above
   the limit the wafer is measured; below it the wafer skips metrology.

---

## The dashboard

| Tab | What it shows |
|---|---|
| **Live wafer stream** | Predictions with a ±2σ band, true MRR, metrology decisions per wafer, and a breakdown of each prediction into its physics / AI / bias parts |
| **Physics vs AI** | The size of the AI residual against its bound, and a parity plot of physics-only vs the full twin |
| **Pad state** | The latent k<sub>p</sub> trajectory across dresser changes, and the learned wear law checked against unseen wafers |
| **Metrology savings** | Error vs. % of wafers skipped (twin-guided vs random sampling), plus an editable cost calculator |
| **What-if recipe** | Change pressure, speed and consumable age, and the twin answers live |

Sidebar controls: playback speed, the σ policy limit, the tool mode (A123 / A456 / B456), and overlays for the GBM
baseline and the Preston term.

---

## Results on the deployment stream

The twin is trained on the first 1,186 labelled wafers (up to day 40.7). It is evaluated on the next 791 (day 40.7–65).

| Model (RMSE, nm/min) | No assimilation | With assimilation* |
|---|---|---|
| Hybrid twin | 5.33 | **4.18** |
| Preston + pad-state ODE only | 5.31 | – |
| Black-box GBM | 7.82 | 6.59 |

\* Assimilation figures assume every wafer is measured.

Raising the σ limit to 5.0 lets **62%** of wafers skip metrology, with an RMSE of 4.42 nm/min on those wafers.

---

## Repository layout

```
app.py                  Streamlit app (5 tabs)
twin_core.py            Active metrology loop, Bayesian bias assimilation, numpy forward pass of the ensemble
generate_demo_data.py   Rebuilds demo_data/ from the raw dataset (needs torch + scikit-learn)
demo_data/
  stream.csv            Per-wafer predictions for the 791 deployment wafers
  history.csv           Training-period wafers, used as the first "past metrology"
  twin_weights.npz      5-member ensemble exported to numpy
  meta.json             Features, scaler, assimilation settings, per-mode stats
.streamlit/config.toml  Theme
demo_backup.mp4         Screen recording of a full replay
```

### Regenerating `demo_data/`

```bash
pip install torch scikit-learn
python generate_demo_data.py "/path/to/2016 PHM DATA CHALLENGE CMP DATA SET/"
```

This takes about 10–15 minutes on a laptop CPU. At the end it checks that the numpy forward pass matches the torch
model (maximum difference about 5e-5 nm/min).

**Design note:** all consumable usage counters (`u_dresser, u_table, u_film, u_membrane, u_sheet, u_dtable, day`)
are masked out of the residual network, so they can only act through the pad-state ODE. If only `day` and
`u_dtable` are masked, the RMSE on future wafers degrades to 6.7.

---

## Things to try

- **Worn-dresser extrapolation.** Turn on *Compare with black-box GBM* and play through days 40.7–46, which run on
  a dresser near end of life. The GBM under-predicts by about 8 nm/min, while the twin stays at about 4.4. At the
  dresser change (orange line, around day 46), the twin's k<sub>p</sub> resets.
- **Metrology trade-off.** Raise the σ limit from 4.4 to 5.0: the share of skipped wafers goes from 42% to 62%, and
  twin-guided sampling stays below random sampling.
- **Tool modes.** B456 has the strongest drift.

The cost figures in the savings tab are illustrative assumptions and can be edited.

---

**Data:** [PHM Society 2016 Data Challenge](https://www.phmsociety.org/) (CMP dataset).
The raw data is not included in this repository.
