"""Core logic for the live dashboard: load exported data, run the active metrology loop, numpy forward pass of the twin.
No torch needed - the ensemble weights were exported to numpy by generate_demo_data.py."""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'demo_data')


def load():
    stream = pd.read_csv(os.path.join(DATA, 'stream.csv')).sort_values('day').reset_index(drop=True)
    history = pd.read_csv(os.path.join(DATA, 'history.csv'))
    meta = json.load(open(os.path.join(DATA, 'meta.json')))
    W = dict(np.load(os.path.join(DATA, 'twin_weights.npz')))
    # dresser change = dresser usage counter drops between consecutive wafers of the same mode
    stream['dresser_change'] = False
    for m, g in stream.groupby('mode'):
        stream.loc[g.index, 'dresser_change'] = g.u_dresser.diff().fillna(0).values < -50
    return stream, history, meta, W


# ---------------------------------------------------------------------------------------------------------------
# Active twin decision loop
# ---------------------------------------------------------------------------------------------------------------
def run_loop(stream, history, meta, threshold=None, measure_mask=None, K=40):
    """Replays the deployment stream in time order.

    For every wafer the twin (1) predicts MRR, (2) corrects it with a Bayesian bias estimated from the most recent
    metrology results of the same mode, (3) computes its total uncertainty sigma, and (4) sends the wafer to
    physical metrology if sigma > threshold. Measured wafers feed the next bias update.
    The GBM baseline gets exactly the same metrology results for its own bias correction.
    Pass `measure_mask` instead of `threshold` to force a given sampling plan (used for the random baseline)."""
    tau_t, kap_t = meta['assim']['twin']['tau'], meta['assim']['twin']['kappa']
    tau_g, kap_g = meta['assim']['gbm']['tau'], meta['assim']['gbm']['kappa']
    s2, scale = meta['sig_e2'], meta['sigma_scale']
    n = len(stream)
    mode = stream['mode'].values; day = stream.day.values; day_end = stream.day_end.values
    y = stream.MRR_true.values; mu = stream.mu.values; gbm = stream.gbm.values
    var_model = stream.epi_var.values + stream.ale_var.values

    # metrology history per mode: (day_end, twin error, gbm error) - starts with the training-period measurements
    H = {m: [list(g.day_end.values), list(g.e_twin.values), list(g.e_gbm.values)]
         for m, g in history.sort_values('day_end').groupby('mode')}
    pending = []   # measured wafers whose result is not yet available (still polishing)

    b_t = np.zeros(n); b_g = np.zeros(n); sig = np.zeros(n); meas = np.zeros(n, bool)
    for i in range(n):
        # release metrology results that are now available
        if pending:
            keep = []
            for j in pending:
                if day_end[j] <= day[i]:
                    h = H[mode[j]]; h[0].append(day_end[j]); h[1].append(y[j] - mu[j]); h[2].append(y[j] - gbm[j])
                else:
                    keep.append(j)
            pending = keep
        h = H[mode[i]]
        t_ = np.asarray(h[0][-K:]); w = np.exp(-(day[i] - t_) / tau_t); Wt = w.sum()
        b_t[i] = (w * np.asarray(h[1][-K:])).sum() / (Wt + kap_t)
        wg = np.exp(-(day[i] - t_) / tau_g)
        b_g[i] = (wg * np.asarray(h[2][-K:])).sum() / (wg.sum() + kap_g)
        sig[i] = scale * np.sqrt(var_model[i] + s2[mode[i]] / (Wt + kap_t))
        meas[i] = (sig[i] > threshold) if measure_mask is None else measure_mask[i]
        if meas[i]:
            pending.append(i)
    out = stream.copy()
    out['bias'] = b_t; out['pred'] = mu + b_t; out['sigma'] = sig; out['measured'] = meas
    out['gbm_pred'] = gbm + b_g
    out['err'] = out.pred - y; out['gbm_err'] = out.gbm_pred - y
    return out


def skipped_stats(res, upto=None):
    r = res if upto is None else res.iloc[:upto]
    sk = r[~r.measured]
    if len(sk) == 0:
        return dict(n=len(r), n_meas=int(r.measured.sum()), n_skip=0, rmse=np.nan, p95=np.nan, max=np.nan,
                    gbm_rmse=np.nan, gbm_max=np.nan)
    e, g = sk.err.abs().values, sk.gbm_err.abs().values
    return dict(n=len(r), n_meas=int(r.measured.sum()), n_skip=len(sk), rmse=float(np.sqrt(np.mean(e ** 2))),
                p95=float(np.quantile(e, .95)), max=float(e.max()),
                gbm_rmse=float(np.sqrt(np.mean(g ** 2))), gbm_max=float(g.max()))


def tradeoff(stream, history, meta, thresholds, n_random=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for th in thresholds:
        r = run_loop(stream, history, meta, threshold=th)
        s = skipped_stats(r)
        frac = r.measured.mean()
        rr = []
        for _ in range(n_random):
            mask = rng.random(len(stream)) < frac
            rr.append(skipped_stats(run_loop(stream, history, meta, measure_mask=mask)))
        rows.append(dict(threshold=th, skip_pct=100 * (1 - frac), rmse_twin=s['rmse'], p95_twin=s['p95'],
                         max_twin=s['max'], rmse_random=np.nanmean([x['rmse'] for x in rr]),
                         p95_random=np.nanmean([x['p95'] for x in rr]), max_random=np.nanmean([x['max'] for x in rr])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Numpy forward pass of the 5-member twin (for the what-if panel)
# ---------------------------------------------------------------------------------------------------------------
def silu(a):
    return a / (1 + np.exp(-a))


def twin_forward(W, meta, mode, feats):
    """feats: dict FEAT -> array (same length). Returns dict of arrays (ensemble mean / std)."""
    FEATS, MODES = meta['FEATS'], meta['MODES']
    m = MODES.index(mode)
    n = max(len(np.atleast_1d(feats[f])) for f in FEATS)
    X = np.column_stack([np.broadcast_to(np.asarray(feats[f], float), (n,)) for f in FEATS])
    xs = (X - np.array(meta['scaler_mean'])) / np.array(meta['scaler_scale'])
    xs = np.hstack([xs, np.tile(np.eye(len(MODES))[m], (n, 1))])
    Pb, Vb = meta['P_bar'][m], meta['V_bar'][m]
    P = np.maximum(X[:, FEATS.index('P')], 0.05 * Pb); V = np.maximum(X[:, FEATS.index('V')], 0.05 * Vb)
    d = X[:, FEATS.index('u_dresser')] / 800.0; u = X[:, FEATS.index('u_table')] / 400.0
    g = (X[:, FEATS.index('u_dtable')] - meta['g_lo']) / (meta['g_hi'] - meta['g_lo'])
    mus, phys, res, ales, kps = [], [], [], [], []
    for k in range(meta['n_ensemble']):
        z_d = (1 - W[f'{k}_amp_d'][m]) + W[f'{k}_amp_d'][m] * np.exp(-W[f'{k}_lam_d'][m] * d)
        z_p = (1 - W[f'{k}_amp_p'][m]) + W[f'{k}_amp_p'][m] * np.exp(-W[f'{k}_lam_p'][m] * u)
        z_g = 1 - W[f'{k}_amp_g'][m] * np.exp(-W[f'{k}_lam_g'][m] * g)
        kp = W[f'{k}_k'][m] * z_d * z_p * z_g
        ph = kp * (P / Pb) ** W[f'{k}_alpha'][m] * (V / Vb) ** W[f'{k}_beta'][m]
        h = xs * W[f'{k}_mask']
        h = silu(h @ W[f'{k}_l0_w'].T + W[f'{k}_l0_b'])
        h = silu(h @ W[f'{k}_l1_w'].T + W[f'{k}_l1_b'])
        h = h @ W[f'{k}_l2_w'].T + W[f'{k}_l2_b']
        r = W[f'{k}_bound'][m] * np.tanh(h[:, 0])
        mus.append(ph + r); phys.append(ph); res.append(r); ales.append(np.exp(np.clip(h[:, 1], -6, 6))); kps.append(kp)
    mus = np.array(mus)
    sigma = meta['sigma_scale'] * np.sqrt(mus.var(0) + np.mean(ales, 0))
    return dict(mu=mus.mean(0), phys=np.mean(phys, 0), resid=np.mean(res, 0), sigma=sigma, kp=np.mean(kps, 0))
