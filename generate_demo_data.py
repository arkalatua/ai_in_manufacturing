"""
generate_demo_data.py  -  trains the twin and builds everything the Streamlit dashboard needs.

Uses a chronological "deployment" set-up:
    * first 60 % of labelled wafers in time  -> training history (what the fab has already measured)
    * last 40 %  (day ~40.7 -> 65)            -> deployment stream replayed live in the dashboard

Outputs (in ./demo_data):
    stream.csv        one row per deployment wafer: truth, twin mean, uncertainty, Preston term, AI residual,
                      latent pad-state, GBM baseline prediction, consumable clocks
    history.csv       training-period wafers (MRR + dresser usage) and their out-of-fold twin errors,
                      used as the first "past metrology" the twin assimilates
    twin_weights.npz  the 5-member twin ensemble exported to plain numpy (the app needs no torch)
    meta.json         feature list, scaler, assimilation settings, per-mode stats

Usage:   python generate_demo_data.py "/path/to/2016 PHM DATA CHALLENGE CMP DATA SET/"
Runtime: ~10-15 min on a laptop CPU (5 ensemble members + 5-fold OOF for the assimilation tuning).
"""
import os, sys, glob, json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
torch.set_num_threads(max(1, os.cpu_count() or 1))

ROOT = sys.argv[1] if len(sys.argv) > 1 else '2016 PHM DATA CHALLENGE CMP DATA SET/'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_data')
os.makedirs(OUT, exist_ok=True)

CFG = dict(seed=0, outlier_mrr=1000.0, resid_bound=0.15, resid_reg=1.0, n_ensemble=int(os.environ.get('N_ENS', 5)), n_folds=5, cv_seeds=2,
           epochs_phys=1500, epochs_joint=2500, metrology_lag_h=0.0, deploy_frac=0.40)
MODES = ['A123', 'A456', 'B456']


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


set_seed(CFG['seed'])

# ----------------------------------------------------------------------------------------------------------------
# Load raw data + wafer-level features
# ----------------------------------------------------------------------------------------------------------------
def load_raw(split):
    files = sorted(glob.glob(os.path.join(ROOT, 'CMP-data', split, '*.csv')))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


SHORT = {
    'USAGE_OF_BACKING_FILM': 'u_film', 'USAGE_OF_DRESSER': 'u_dresser', 'USAGE_OF_POLISHING_TABLE': 'u_table',
    'USAGE_OF_DRESSER_TABLE': 'u_dtable', 'USAGE_OF_MEMBRANE': 'u_membrane', 'USAGE_OF_PRESSURIZED_SHEET': 'u_sheet',
    'PRESSURIZED_CHAMBER_PRESSURE': 'p_chamber', 'MAIN_OUTER_AIR_BAG_PRESSURE': 'p_main_outer',
    'CENTER_AIR_BAG_PRESSURE': 'p_center', 'RETAINER_RING_PRESSURE': 'p_retainer',
    'RIPPLE_AIR_BAG_PRESSURE': 'p_ripple', 'EDGE_AIR_BAG_PRESSURE': 'p_edge',
    'SLURRY_FLOW_LINE_A': 'slurry_a', 'SLURRY_FLOW_LINE_B': 'slurry_b', 'SLURRY_FLOW_LINE_C': 'slurry_c',
    'WAFER_ROTATION': 'rot_wafer', 'STAGE_ROTATION': 'rot_stage', 'HEAD_ROTATION': 'rot_head',
}
KEY = ['WAFER_ID', 'STAGE']


def wafer_features(raw):
    df = raw.rename(columns=SHORT).sort_values(KEY + ['TIMESTAMP']).reset_index(drop=True)
    df['dt'] = df.groupby(KEY).TIMESTAMP.diff().fillna(0).clip(0, 10)
    df['P'], df['V'] = df.p_chamber, df.rot_stage
    df['PV'] = df.P * df.V
    df['work'] = df.PV * df.dt
    g = df.groupby(KEY)
    base = g.agg(machine_data=('MACHINE_DATA', 'first'), t_start=('TIMESTAMP', 'min'), t_end=('TIMESTAMP', 'max'),
                 n_rows=('TIMESTAMP', 'size'), work=('work', 'sum'), dressing_water=('DRESSING_WATER_STATUS', 'mean'),
                 u_film=('u_film', 'first'), u_dresser=('u_dresser', 'first'), u_table=('u_table', 'first'),
                 u_dtable=('u_dtable', 'first'), u_membrane=('u_membrane', 'first'), u_sheet=('u_sheet', 'first'))
    act = df[df.V > 0]
    ga = act.groupby(KEY)
    cols = ['P', 'V', 'PV', 'rot_head', 'rot_wafer', 'p_main_outer', 'p_center', 'p_edge', 'p_retainer', 'p_ripple',
            'slurry_a', 'slurry_b', 'slurry_c']
    am = ga[cols].mean()
    am['P_std'], am['V_std'], am['active_time'] = ga.P.std(), ga.V.std(), ga.dt.sum()
    f = base.join(am, how='left').reset_index()
    f['duration'] = f.t_end - f.t_start
    f['mode'] = f.STAGE + np.where(f.machine_data == 1, '123', '456')
    return f


t_all = time.time()
print('loading raw sensor data ...')
raw_tr, raw_te = load_raw('training'), load_raw('test')
lab_tr = pd.read_csv(os.path.join(ROOT, 'CMP-training-removalrate.csv'))
lab_all = wafer_features(raw_tr).merge(lab_tr, on=KEY, how='left').rename(columns={'AVG_REMOVAL_RATE': 'MRR'})
fsub = wafer_features(raw_te)
lab_all['MRR'] = lab_all.MRR.astype(float)
lab_all = lab_all[lab_all.MRR <= CFG['outlier_mrr']].reset_index(drop=True)

T0 = min(lab_all.t_start.min(), fsub.t_start.min())            # common time origin for all wafers
for f in (lab_all,):
    f['day'] = (f.t_start - T0) / 86400.0
    f['day_end'] = (f.t_end - T0) / 86400.0
    f['mode_id'] = f['mode'].map({m: i for i, m in enumerate(MODES)}).astype(int)

# Chronological deployment split: the twin is trained on the past and replayed on the future
lab_all = lab_all.sort_values('t_start').reset_index(drop=True)
cut = lab_all.day.quantile(1 - CFG['deploy_frac'])
ftr = lab_all[lab_all.day < cut].reset_index(drop=True)
fdep = lab_all[lab_all.day >= cut].reset_index(drop=True)
print(f'train (history) {len(ftr)} wafers, day 0-{cut:.1f} | deployment stream {len(fdep)} wafers, day {cut:.1f}-{fdep.day.max():.1f}')

FEATS = ['P', 'V', 'PV', 'P_std', 'V_std', 'rot_head', 'rot_wafer', 'p_main_outer', 'p_center', 'p_edge', 'p_retainer',
         'p_ripple', 'slurry_a', 'slurry_b', 'slurry_c', 'active_time', 'duration', 'n_rows', 'work', 'dressing_water',
         'u_film', 'u_dresser', 'u_table', 'u_dtable', 'u_membrane', 'u_sheet', 'day']
CLOCKS = os.environ.get('CLOCKS', 'day,u_dtable,u_film,u_membrane,u_sheet,u_dresser,u_table').split(',')   # monotone clocks: kept out of the residual net
for f in (ftr, fdep):
    f[FEATS] = f[FEATS].astype(float)
med = ftr.groupby('mode')[FEATS].median()
for f in (ftr, fdep):
    for m in MODES:
        idx = f['mode'] == m
        f.loc[idx, FEATS] = f.loc[idx, FEATS].fillna(med.loc[m])
    f[FEATS] = f[FEATS].astype(float)


# ----------------------------------------------------------------------------------------------------------------
# Physics-informed hybrid twin
# ----------------------------------------------------------------------------------------------------------------
class Prep:
    def fit(self, df):
        self.sc = StandardScaler().fit(df[FEATS])
        self.P_bar = df.groupby('mode_id').P.mean().reindex(range(len(MODES))).values
        self.V_bar = df.groupby('mode_id').V.mean().reindex(range(len(MODES))).values
        self.mrr_mean = df.groupby('mode_id').MRR.mean().reindex(range(len(MODES))).values
        self.g_lo, self.g_hi = df.u_dtable.min(), df.u_dtable.max()
        return self

    def __call__(self, df):
        m = df.mode_id.values
        x = np.hstack([self.sc.transform(df[FEATS]), np.eye(len(MODES))[m]])
        P = np.maximum(df.P.values, 0.05 * self.P_bar[m]); V = np.maximum(df.V.values, 0.05 * self.V_bar[m])
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        return dict(x=t(x), m=torch.tensor(m, dtype=torch.long),
                    d=t(df.u_dresser.values / 800.0), u=t(df.u_table.values / 400.0),
                    g=t((df.u_dtable.values - self.g_lo) / (self.g_hi - self.g_lo)),
                    lp=t(np.log(P / self.P_bar[m])), lv=t(np.log(V / self.V_bar[m])))


class HybridTwin(nn.Module):
    def __init__(self, n_in, mrr_mean, hidden=64, bound=0.15, use_ode=True, use_pv=True, use_resid=True, mask_clocks=True):
        super().__init__()
        M = len(MODES)
        mask = torch.ones(n_in)
        if mask_clocks: mask[[FEATS.index(c) for c in CLOCKS]] = 0.0
        self.register_buffer('in_mask', mask)
        self.use_ode, self.use_pv, self.use_resid = use_ode, use_pv, use_resid
        self.log_k = nn.Parameter(torch.log(torch.tensor(mrr_mean, dtype=torch.float32)))
        self.raw_alpha = nn.Parameter(torch.full((M,), -2.0)); self.raw_beta = nn.Parameter(torch.full((M,), -2.0))
        self.raw_lam_d = nn.Parameter(torch.zeros(M)); self.raw_amp_d = nn.Parameter(torch.full((M,), -1.0))
        self.raw_lam_p = nn.Parameter(torch.zeros(M)); self.raw_amp_p = nn.Parameter(torch.full((M,), -3.0))
        self.raw_lam_g = nn.Parameter(torch.zeros(M)); self.raw_amp_g = nn.Parameter(torch.full((M,), -1.0))
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.register_buffer('bound', bound * torch.tensor(mrr_mean, dtype=torch.float32))

    def physics_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith('net.')]

    def ode(self, m, d, u, g):
        lam_d, amp_d = 3 * F.softplus(self.raw_lam_d[m]), torch.sigmoid(self.raw_amp_d[m])
        lam_p, amp_p = 3 * F.softplus(self.raw_lam_p[m]), 0.5 * torch.sigmoid(self.raw_amp_p[m])
        z_d = (1 - amp_d) + amp_d * torch.exp(-lam_d * d)
        z_p = (1 - amp_p) + amp_p * torch.exp(-lam_p * u)
        lam_g, amp_g = 3 * F.softplus(self.raw_lam_g[m]), 0.5 * torch.sigmoid(self.raw_amp_g[m])
        z_g = 1 - amp_g * torch.exp(-lam_g * g)
        return z_d, z_p, z_g

    def forward(self, b):
        m = b['m']
        log_kp = self.log_k[m]
        if self.use_ode:
            z_d, z_p, z_g = self.ode(m, b['d'], b['u'], b['g'])
            log_kp = log_kp + torch.log(z_d) + torch.log(z_p) + torch.log(z_g)
        log_phys = log_kp
        if self.use_pv:
            log_phys = log_phys + 2 * torch.sigmoid(self.raw_alpha[m]) * b['lp'] + 2 * torch.sigmoid(self.raw_beta[m]) * b['lv']
        phys = torch.exp(log_phys)
        h = self.net(b['x'] * self.in_mask)
        resid = self.bound[m] * torch.tanh(h[:, 0]) if self.use_resid else torch.zeros_like(phys)
        logvar = torch.clamp(h[:, 1], -6, 6)
        return phys + resid, logvar, phys, resid, torch.exp(log_kp)


def _sub(b, idx):
    return {k: v[idx] for k, v in b.items()}


def train_twin(df_tr, prep, seed=0, kind='hybrid', val_frac=0.15, verbose=False):
    set_seed(seed)
    b = prep(df_tr); y = torch.tensor(df_tr.MRR.values, dtype=torch.float32)
    n = len(y); perm = np.random.permutation(n); nv = int(val_frac * n)
    iv, it = torch.tensor(perm[:nv]), torch.tensor(perm[nv:])
    cfg = dict(preston=dict(use_ode=False, use_pv=True, use_resid=False),
               physics=dict(use_ode=True, use_pv=True, use_resid=False),
               hybrid=dict(use_ode=True, use_pv=True, use_resid=True),
               blackbox=dict(use_ode=False, use_pv=False, use_resid=True))[kind]
    bound = 1.0 if kind == 'blackbox' else CFG['resid_bound']
    model = HybridTwin(b['x'].shape[1], prep.mrr_mean, bound=bound, mask_clocks=(kind != 'blackbox'), **cfg)
    bt, bv, yt, yv = _sub(b, it), _sub(b, iv), y[it], y[iv]
    if kind != 'blackbox':
        net_flag = model.use_resid; model.use_resid = False
        opt = torch.optim.Adam(model.physics_params(), lr=0.03)
        for ep in range(CFG['epochs_phys']):
            opt.zero_grad(); mu = model(bt)[0]; loss = F.mse_loss(mu, yt); loss.backward(); opt.step()
        model.use_resid = net_flag
    opt = torch.optim.Adam([{'params': model.physics_params(), 'lr': 3e-3},
                            {'params': model.net.parameters(), 'lr': 3e-3, 'weight_decay': 1e-4}])
    best, best_state, bad = np.inf, None, 0
    for ep in range(CFG['epochs_joint']):
        model.train(); opt.zero_grad()
        mu, logvar, phys, resid, _ = model(bt)
        mse = F.mse_loss(mu, yt)
        nll = 0.5 * (logvar + (yt - mu.detach()) ** 2 / torch.exp(logvar)).mean()
        reg = ((resid / model.bound[bt['m']]) ** 2).mean() if model.use_resid and kind != 'blackbox' else 0.0
        loss = mse + 0.1 * nll + CFG['resid_reg'] * reg
        loss.backward(); opt.step()
        if ep % 25 == 0:
            model.eval()
            with torch.no_grad():
                v = F.mse_loss(model(bv)[0], yv).item()
            if v < best - 1e-4: best, bad, best_state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
            else: bad += 1
            if bad > 30: break
    model.load_state_dict(best_state); model.eval()
    if verbose: print(f'  {kind} seed {seed}: val RMSE {np.sqrt(best):.3f}  (stopped ep {ep})')
    return model


@torch.no_grad()
def predict(models, prep, df):
    b = prep(df)
    outs = [m(b) for m in models]
    mu = torch.stack([o[0] for o in outs]).numpy()
    ale = torch.stack([torch.exp(o[1]) for o in outs]).numpy()
    zs = [m.ode(b['m'], b['d'], b['u'], b['g']) for m in models]
    return dict(mu=mu.mean(0), epi_var=mu.var(0), ale_var=ale.mean(0),
                phys=torch.stack([o[2] for o in outs]).numpy().mean(0),
                resid=torch.stack([o[3] for o in outs]).numpy().mean(0),
                kp=torch.stack([o[4] for o in outs]).numpy().mean(0),
                z_d=np.mean([z[0].numpy() for z in zs], 0), z_p=np.mean([z[1].numpy() for z in zs], 0),
                z_g=np.mean([z[2].numpy() for z in zs], 0))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def make_gbm(seed=0):
    return HistGradientBoostingRegressor(max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=10,
                                         l2_regularization=1.0, random_state=seed)


# ----------------------------------------------------------------------------------------------------------------
# Train the deployment twin + GBM baseline on the history
# ----------------------------------------------------------------------------------------------------------------
t0 = time.time()
prep = Prep().fit(ftr)
print('training 5-member twin ensemble ...')
twin = [train_twin(ftr, prep, seed=s, kind='hybrid', verbose=True) for s in range(CFG['n_ensemble'])]
print('training physics-only twin (Preston + pad-state ODE) ...')
phys_only = [train_twin(ftr, prep, seed=s, kind='physics') for s in range(3)]
Xtr = np.hstack([ftr[FEATS].values, np.eye(3)[ftr.mode_id]]); Xdep = np.hstack([fdep[FEATS].values, np.eye(3)[fdep.mode_id]])
gbm = make_gbm().fit(Xtr, ftr.MRR)
print(f'trained in {time.time() - t0:.0f}s')

pd_ = predict(twin, prep, fdep)
gbm_dep = gbm.predict(Xdep)
phys_dep = predict(phys_only, prep, fdep)['mu']
print(f'deployment RMSE (no assimilation): twin {rmse(pd_["mu"], fdep.MRR):.3f} | GBM {rmse(gbm_dep, fdep.MRR):.3f} '
      f'| physics-only {rmse(phys_dep, fdep.MRR):.3f}')
if os.environ.get('QUICK'):
    sys.exit(0)

# ----------------------------------------------------------------------------------------------------------------
# Out-of-fold errors on the history (honest "past metrology" for the assimilation step) + tuning tau, kappa
# ----------------------------------------------------------------------------------------------------------------
print('out-of-fold predictions for assimilation tuning ...')
kf = KFold(CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
oof_twin, oof_gbm, oof_sig2 = np.zeros(len(ftr)), np.zeros(len(ftr)), np.zeros(len(ftr))
for fold, (i_tr, i_va) in enumerate(kf.split(ftr)):
    a, v = ftr.iloc[i_tr].reset_index(drop=True), ftr.iloc[i_va].reset_index(drop=True)
    pp = Prep().fit(a)
    ms = [train_twin(a, pp, seed=100 * fold + s, kind='hybrid') for s in range(CFG['cv_seeds'])]
    pv = predict(ms, pp, v)
    oof_twin[i_va] = pv['mu']; oof_sig2[i_va] = pv['epi_var'] + pv['ale_var']
    oof_gbm[i_va] = make_gbm(fold).fit(Xtr[i_tr], ftr.MRR.values[i_tr]).predict(Xtr[i_va])
    print(f'  fold {fold}: twin OOF RMSE {rmse(oof_twin[i_va], v.MRR):.3f}  ({time.time() - t0:.0f}s)')

LAG = CFG['metrology_lag_h'] / 24.0


def assimilate(tgt, hist, tau, kappa, K=40):
    b, W = np.zeros(len(tgt)), np.zeros(len(tgt))
    for m in MODES:
        h = hist[hist['mode'] == m].sort_values('day_end'); ht, he = h.day_end.values.astype(float), h.e.values.astype(float)
        it = np.where(tgt['mode'].values == m)[0]
        tday = tgt.day.values.astype(float)
        pos = np.searchsorted(ht, tday[it] - LAG, side='left')
        for i, p in zip(it, pos):
            lo = max(0, p - K); dt_ = tday[i] - ht[lo:p]
            w = np.exp(-dt_ / tau); W[i] = w.sum(); b[i] = (w * he[lo:p]).sum() / (W[i] + kappa)
    return b, W


def tune(oof_pred):
    hist = pd.DataFrame({'mode': ftr['mode'], 'day_end': ftr.day_end, 'e': ftr.MRR - oof_pred})
    best = None
    for tau in [1 / 24, 3 / 24, 6 / 24, 12 / 24, 1, 2, 4]:
        for kappa in [0.25, 0.5, 1, 2, 4, 8]:
            b, W = assimilate(ftr, hist, tau, kappa)
            r = rmse(oof_pred + b, ftr.MRR)
            if best is None or r < best[0]: best = (r, tau, kappa, b, W)
    return best


r_t, tau_t, kap_t, b_t, W_t = tune(oof_twin)
r_g, tau_g, kap_g, _, _ = tune(oof_gbm)
print(f'assimilation: twin tau={tau_t * 24:.1f}h kappa={kap_t} (OOF RMSE {r_t:.3f}) | GBM tau={tau_g * 24:.1f}h kappa={kap_g} (OOF {r_g:.3f})')

# Noise level of a single metrology "innovation" e = y - mu, per mode (after assimilation) -> bias posterior variance
sig_e2 = {m: float(np.var((ftr.MRR - oof_twin - b_t)[ftr['mode'] == m])) for m in MODES}
# Calibration: scale the ensemble sigma so that z-scores on OOF have unit variance
z = (ftr.MRR - oof_twin - b_t) / np.sqrt(oof_sig2 + np.array([sig_e2[m] for m in ftr['mode']]) / (W_t + kap_t))
sigma_scale = float(np.std(z))
print('sigma_e^2 per mode', {k: round(v, 2) for k, v in sig_e2.items()}, '| sigma calibration scale', round(sigma_scale, 3))

# ----------------------------------------------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------------------------------------------
stream = pd.DataFrame({
    'wafer_id': fdep.WAFER_ID.values, 'stage': fdep.STAGE.values, 'mode': fdep['mode'].values,
    'day': fdep.day.values, 'day_end': fdep.day_end.values, 'MRR_true': fdep.MRR.values,
    'mu': pd_['mu'], 'epi_var': pd_['epi_var'], 'ale_var': pd_['ale_var'],
    'physics_term': pd_['phys'], 'ai_residual': pd_['resid'], 'kp': pd_['kp'],
    'z_dresser': pd_['z_d'], 'z_pad': pd_['z_p'], 'z_breakin': pd_['z_g'],
    'gbm': gbm_dep, 'physics_only': phys_dep,
    'u_dresser': fdep.u_dresser.values, 'u_table': fdep.u_table.values, 'u_dtable': fdep.u_dtable.values,
    'P': fdep.P.values, 'V': fdep.V.values,
})
stream.to_csv(f'{OUT}/stream.csv', index=False)

history = pd.DataFrame({'wafer_id': ftr.WAFER_ID, 'mode': ftr['mode'], 'day': ftr.day, 'day_end': ftr.day_end,
                        'MRR_true': ftr.MRR, 'u_dresser': ftr.u_dresser, 'e_twin': ftr.MRR - oof_twin,
                        'e_gbm': ftr.MRR - oof_gbm})
history.to_csv(f'{OUT}/history.csv', index=False)

# Twin ensemble -> numpy (physics parameters already transformed + MLP weights)
W = {}
for k, m in enumerate(twin):
    sd = {n: p.detach().numpy() for n, p in m.state_dict().items()}
    with torch.no_grad():
        W[f'{k}_k'] = np.exp(sd['log_k'])
        W[f'{k}_alpha'] = 2 / (1 + np.exp(-sd['raw_alpha'])); W[f'{k}_beta'] = 2 / (1 + np.exp(-sd['raw_beta']))
        sp = lambda a: np.log1p(np.exp(a)); sg = lambda a: 1 / (1 + np.exp(-a))
        W[f'{k}_lam_d'], W[f'{k}_amp_d'] = 3 * sp(sd['raw_lam_d']), sg(sd['raw_amp_d'])
        W[f'{k}_lam_p'], W[f'{k}_amp_p'] = 3 * sp(sd['raw_lam_p']), 0.5 * sg(sd['raw_amp_p'])
        W[f'{k}_lam_g'], W[f'{k}_amp_g'] = 3 * sp(sd['raw_lam_g']), 0.5 * sg(sd['raw_amp_g'])
    for li, name in [(0, 'l0'), (3, 'l1'), (5, 'l2')]:
        W[f'{k}_{name}_w'] = sd[f'net.{li}.weight']; W[f'{k}_{name}_b'] = sd[f'net.{li}.bias']
    W[f'{k}_mask'] = sd['in_mask']; W[f'{k}_bound'] = sd['bound']
np.savez(f'{OUT}/twin_weights.npz', **W)

# Typical (median) operating point per mode, used by the what-if panel
ref = {m: {c: float(v) for c, v in ftr[ftr['mode'] == m][FEATS].median().items()} for m in MODES}
rng = {m: {c: [float(ftr[ftr['mode'] == m][c].quantile(.01)), float(ftr[ftr['mode'] == m][c].quantile(.99))]
           for c in ['P', 'V', 'u_dresser', 'u_table']} for m in MODES}
meta = dict(
    FEATS=FEATS, MODES=MODES, n_ensemble=CFG['n_ensemble'], cut_day=float(cut),
    scaler_mean=prep.sc.mean_.tolist(), scaler_scale=prep.sc.scale_.tolist(),
    P_bar=prep.P_bar.tolist(), V_bar=prep.V_bar.tolist(), mrr_mean=prep.mrr_mean.tolist(),
    g_lo=float(prep.g_lo), g_hi=float(prep.g_hi),
    assim=dict(twin=dict(tau=tau_t, kappa=kap_t), gbm=dict(tau=tau_g, kappa=kap_g), K=40),
    sig_e2=sig_e2, sigma_scale=sigma_scale, ref_point=ref, ranges=rng,
    deploy_rmse=dict(twin=rmse(pd_['mu'], fdep.MRR), gbm=rmse(gbm_dep, fdep.MRR), physics_only=rmse(phys_dep, fdep.MRR)),
    n_train=len(ftr), n_deploy=len(fdep),
)
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1)
fdep[FEATS].to_csv(f'{OUT}/stream_features.csv', index=False)

# Self-check: the numpy forward pass used by the app must reproduce the torch ensemble exactly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_core
Wn = dict(np.load(f'{OUT}/twin_weights.npz'))
for m in MODES:
    sel = (fdep['mode'] == m).values
    o = twin_core.twin_forward(Wn, meta, m, {c: fdep.loc[sel, c].values for c in FEATS})
    err = np.abs(o['mu'] - pd_['mu'][sel]).max()
    print(f'numpy vs torch forward pass, mode {m}: max |diff| = {err:.2e} nm/min')
    assert err < 1e-3
print(f'done in {time.time() - t_all:.0f}s -> {OUT}')
