"""
CMP Digital Twin - live demo
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import twin_core as tc

st.set_page_config(page_title='CMP Digital Twin', layout='wide')

# ----------------------------------------------------------------------------------------------------------------
# Look & feel (light "lab instrument" style, large type for a projector)
# ----------------------------------------------------------------------------------------------------------------
BLUE, AQUA, INK, MUTED, GRID, AXIS = '#1f5fbf', '#2a8c7a', '#16181d', '#8a8c91', '#e6e4dd', '#16181d'
RED, GREEN, ORANGE, VIOLET, YELLOW = '#b4332f', '#2f7d3a', '#c65d21', '#5b4bb7', '#b8860b'
SECOND, LINE, BAND = '#5b5e66', '#dcdad3', 'rgba(31,95,191,0.10)'
SANS = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, Consolas, monospace"
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
  :root { --ink:#16181d; --second:#5b5e66; --line:#dcdad3; --panel:#ffffff; --accent:#1f5fbf;
          --sans:'IBM Plex Sans','Segoe UI',system-ui,sans-serif; --mono:'IBM Plex Mono',ui-monospace,Consolas,monospace; }
  html, body, [class*="css"], .stApp, .stMarkdown, button, input, label { font-family: var(--sans); }
  html, body { font-size: 17px; }
  .block-container { padding-top: 1.4rem; padding-bottom: 1rem; max-width: 1500px; }
  #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton { display: none !important; }
  header[data-testid="stHeader"] { background: transparent; }

  /* masthead */
  .mast { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;
          border-bottom: 2px solid var(--ink); padding-bottom: .55rem; }
  .mast-title { font-size: 1.7rem; font-weight: 600; letter-spacing: .01em; line-height: 1.1; }
  .mast-sub { font-size: .95rem; color: var(--second); margin-top: .2rem; }
  .mast-meta { font-family: var(--mono); font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--second); text-align:right; }
  .mast-desc { color: var(--second); font-size: .92rem; max-width: 95ch; margin: .6rem 0 .2rem 0; }

  /* micro-labels, ledes, footnote */
  .lede { color: var(--second); font-size: .98rem; max-width: 82ch; margin: .2rem 0 .8rem 0;
          border-left: 2px solid var(--line); padding-left: .8rem; }
  .section { font-size: .78rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--second);
             border-bottom: 1px solid var(--line); padding-bottom: .35rem; margin: 1.4rem 0 .6rem 0; }
  .foot { font-family: var(--mono); font-size: .74rem; color: var(--second); border-top: 1px solid var(--line);
          padding-top: .6rem; margin-top: 1.4rem; }

  /* tabs: flat underline */
  .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--line); }
  .stTabs [data-baseweb="tab"] { padding: .55rem 0; background: transparent; }
  .stTabs [data-baseweb="tab"] p { font-size: .95rem; font-weight: 500; color: var(--second); }
  .stTabs [aria-selected="true"] p { color: var(--ink); }
  .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); height: 2px; }

  /* metrics: readout with hairline rule, mono value */
  [data-testid="stMetric"] { border-top: 1px solid var(--ink); padding: .45rem 0 .2rem 0; background: transparent; }
  [data-testid="stMetricLabel"] p { font-size: .74rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: var(--second); }
  [data-testid="stMetricValue"] { font-family: var(--mono); font-size: 1.95rem; font-weight: 500; font-variant-numeric: tabular-nums; }
  [data-testid="stMetricDelta"] { font-family: var(--mono); font-size: .85rem; }

  /* sidebar */
  [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
  [data-testid="stSidebar"] h3 { font-size: .76rem !important; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
                                 color: var(--second); border-bottom: 1px solid var(--line); padding: 1rem 0 .35rem 0 !important; margin-bottom: .5rem; }

  /* buttons + inputs */
  .stButton button { border-radius: 3px; border: 1px solid #b9b7af; font-weight: 500; box-shadow: none; }
  .stButton button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
  [data-baseweb="input"], [data-baseweb="select"] > div { border-radius: 3px; }
  [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 2px; }

  /* wafer readout card */
  .wafer-card { border: 1px solid var(--line); border-top: 2px solid var(--ink); background: var(--panel); padding: .8rem 1rem .9rem 1rem; }
  .wc-head { display:flex; justify-content:space-between; font-family: var(--mono); font-size: .8rem; letter-spacing: .06em;
             text-transform: uppercase; color: var(--second); margin-bottom: .5rem; }
  .big { font-family: var(--mono); font-size: 2.2rem; font-weight: 500; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .big small { font-size: 1rem; color: var(--second); }
  .wc-cap { font-size: .8rem; color: var(--second); margin-bottom: .7rem; }
  .status { border-left: 4px solid; padding: .45rem .7rem; font-size: .82rem; font-weight: 600; letter-spacing: .1em;
            text-transform: uppercase; margin-bottom: .7rem; }
  .status span { display:block; font-weight: 400; letter-spacing: 0; text-transform: none; font-size: .85rem; color: var(--second); margin-top: .1rem; }
  .status-meas { border-color: #b4332f; background: #f8ecea; color: #8e2421; }
  .status-skip { border-color: #2f7d3a; background: #ebf3ea; color: #24602d; }
  .kv { display:flex; justify-content:space-between; border-top: 1px dashed var(--line); padding: .3rem 0; font-size: .88rem; }
  .kv b { font-family: var(--mono); font-weight: 500; font-variant-numeric: tabular-nums; }
  .muted { color: var(--second); }
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(template='plotly_white', font=dict(family=SANS, size=14, color=INK),
              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
              margin=dict(l=60, r=20, t=40, b=50),
              hoverlabel=dict(font=dict(family=MONO, size=13, color=INK), bgcolor='#ffffff', bordercolor=LINE),
              legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(size=13, color=SECOND)))


def style(fig, height=430, **kw):
    fig.update_layout(height=height, **{**LAYOUT, **kw})
    fig.update_xaxes(showgrid=False, linecolor=AXIS, linewidth=1, ticks='outside', tickcolor=AXIS, ticklen=4,
                     zeroline=False, tickfont=dict(family=MONO, size=12, color=SECOND), title_font=dict(size=13, color=SECOND))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, linewidth=1, ticks='outside', tickcolor=AXIS, ticklen=4,
                     zeroline=False, tickfont=dict(family=MONO, size=12, color=SECOND), title_font=dict(size=13, color=SECOND))
    return fig


def lede(text):
    st.markdown(f"<div class='lede'>{text}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------------------
# Data + cached simulations
# ----------------------------------------------------------------------------------------------------------------
@st.cache_resource
def load():
    return tc.load()


stream, history, meta, W = load()
N = len(stream)


@st.cache_data(show_spinner=False)
def simulate(threshold):
    return tc.run_loop(stream, history, meta, threshold=threshold)


@st.cache_data(show_spinner='Computing metrology trade-off curve (one-off, a few seconds)...')
def tradeoff_curve():
    lo = np.quantile(simulate(0.0).sigma, 0.03)      # every wafer measured -> smallest achievable sigma
    hi = np.quantile(simulate(1e9).sigma, 0.60)      # nothing measured -> sigma keeps growing
    return tc.tradeoff(stream, history, meta, np.linspace(lo, hi, 12), n_random=8)


# ----------------------------------------------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------------------------------------------
ss = st.session_state
ss.setdefault('idx', 0)          # number of wafers already processed
ss.setdefault('playing', False)

# ----------------------------------------------------------------------------------------------------------------
# Sidebar: controls
# ----------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('### Playback')
    def _step(k):
        ss.idx = min(N, ss.idx + k); ss.playing = False

    def _toggle():
        ss.playing = not ss.playing and ss.idx < N

    def _reset():
        ss.idx = 0; ss.playing = False

    c1, c2 = st.columns(2)
    c1.button('Next wafer', on_click=_step, args=(1,), width='stretch')
    c2.button('+25 wafers', on_click=_step, args=(25,), width='stretch')
    c3, c4 = st.columns(2)
    c3.button('Pause' if ss.playing else 'Play', on_click=_toggle, type='primary', width='stretch')
    c4.button('Reset', on_click=_reset, width='stretch')
    speed = st.select_slider('Play speed (wafers per tick)', [1, 2, 4, 8, 16], value=4)
    jump = st.slider('Jump to wafer #', 0, N, ss.idx, key=f'jump_{ss.idx}')
    if jump != ss.idx:
        ss.idx = jump; ss.playing = False

    st.markdown('### Metrology policy')
    thr = st.slider('Send wafer to metrology if twin uncertainty σ exceeds (nm/min)', 2.8, 7.0, 4.4, 0.1)
    res_full = simulate(thr)
    st.caption(f'Over the full stream this policy measures **{100 * res_full.measured.mean():.0f}%** of wafers '
               f'and lets **{100 * (1 - res_full.measured.mean()):.0f}%** skip metrology.')

    st.markdown('### Chart')
    chart_mode = st.radio('Tool mode shown', meta['MODES'], index=2, horizontal=True,
                          help='A123 = stage A on chambers 1-3, A456 / B456 = stage A / B on chambers 4-6')
    window = st.slider('Visible window (days)', 2, 25, 6)
    show_gbm = st.toggle('Overlay black-box GBM baseline', value=False)
    show_phys = st.toggle('Show Preston physics backbone', value=True)

res = res_full
st.markdown(f"""<div class='mast'>
  <div><div class='mast-title'>CMP Digital Twin</div>
       <div class='mast-sub'>Physics-informed MRR prediction · live fab replay</div></div>
  <div class='mast-meta'>ME6324 · Group 37<br>PHM 2016 CMP data</div></div>
<div class='mast-desc'>Wafers polished between day {meta['cut_day']:.1f} and day {stream.day.max():.0f} are replayed in
time order. The twin was trained only on the {meta['n_train']:,} wafers before that and has never seen these {N} wafers.</div>""",
            unsafe_allow_html=True)

tab_live, tab_phys, tab_pad, tab_trade, tab_what = st.tabs(
    ['Live wafer stream', 'Physics vs AI', 'Pad state', 'Metrology savings', 'What-if recipe'])


# ----------------------------------------------------------------------------------------------------------------
# Tab 1: live stream (fragment so auto-play only refreshes this part)
# ----------------------------------------------------------------------------------------------------------------
def waterfall(row, height=300):
    """How the twin builds this wafer's prediction: physics + bounded AI residual + metrology bias."""
    steps = [('Preston physics', None, row.physics_term, AQUA), ('AI residual', row.physics_term, row.ai_residual, VIOLET),
             ('Metrology bias', row.physics_term + row.ai_residual, row.bias, YELLOW), ('Twin prediction', None, row.pred, BLUE)]
    lo = min(row.physics_term, row.pred, row.MRR_true, row.physics_term + row.ai_residual) - 6
    hi = max(row.physics_term, row.pred, row.MRR_true, row.physics_term + row.ai_residual) + 6
    fig = go.Figure()
    for name, base, val, col in steps:
        b = lo if base is None else base
        w = val - lo if base is None else val
        txt = f'{val:.1f}' if base is None else f'{val:+.1f}'
        fig.add_trace(go.Bar(y=[name], x=[w], base=[b], orientation='h', marker=dict(color=col), text=[txt],
                             textposition='outside', textfont=dict(size=15, color=INK), cliponaxis=False,
                             hovertemplate=f'{name}: {txt} nm/min<extra></extra>'))
    fig.add_vline(x=row.MRR_true, line=dict(color=INK, dash='dot', width=2))
    fig.add_annotation(x=row.MRR_true, y=1.0, yref='paper', yanchor='bottom', showarrow=False,
                       text=f'true {row.MRR_true:.1f}', font=dict(size=13, color=INK))
    style(fig, height=height, showlegend=False, margin=dict(l=10, r=40, t=28, b=40), bargap=0.35)
    fig.update_yaxes(autorange='reversed', gridcolor='rgba(0,0,0,0)', ticks='', tickfont=dict(family=SANS, size=13, color=INK))
    fig.update_xaxes(range=[lo, hi + 3], title='nm/min')
    return fig


def stream_chart(r, mode, cur):
    d = r[r['mode'] == mode]
    t1 = cur.day if cur is not None else meta['cut_day']
    d = d[d.day >= t1 - window]
    fig = go.Figure()
    if len(d):
        x = d.day
        fig.add_trace(go.Scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([d.pred + 2 * d.sigma, (d.pred - 2 * d.sigma)[::-1]]),
                                 fill='toself', fillcolor=BAND, line=dict(width=0),
                                 hoverinfo='skip', name='twin ± 2σ'))
        if show_phys:
            fig.add_trace(go.Scatter(x=x, y=d.physics_term, mode='lines', line=dict(color=AQUA, width=2),
                                     name='Preston physics backbone',
                                     hovertemplate='physics %{y:.1f} nm/min<extra></extra>'))
        if show_gbm:
            fig.add_trace(go.Scatter(x=x, y=d.gbm_pred, mode='lines', line=dict(color=MUTED, width=2, dash='dash'),
                                     name='black-box GBM', hovertemplate='GBM %{y:.1f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=x, y=d.pred, mode='lines', line=dict(color=BLUE, width=2.5), name='twin prediction',
                                 hovertemplate='twin %{y:.1f} nm/min<extra></extra>'))
        sk, me = d[~d.measured], d[d.measured]
        fig.add_trace(go.Scatter(x=sk.day, y=sk.MRR_true, mode='markers', marker=dict(color=INK, size=8),
                                 name='true MRR (metrology skipped)', customdata=np.c_[sk.wafer_id, sk.err],
                                 hovertemplate='wafer %{customdata[0]}<br>true %{y:.1f}<br>twin error %{customdata[1]:+.1f}<extra>skipped</extra>'))
        fig.add_trace(go.Scatter(x=me.day, y=me.MRR_true, mode='markers',
                                 marker=dict(color=RED, size=11, symbol='x-thin', line=dict(width=2.5, color=RED)),
                                 name='sent to metrology', customdata=np.c_[me.wafer_id, me.sigma],
                                 hovertemplate='wafer %{customdata[0]}<br>measured %{y:.1f}<br>σ was %{customdata[1]:.2f}<extra>metrology</extra>'))
        for dc in d[d.dresser_change].day:
            fig.add_vline(x=dc, line=dict(color=ORANGE, width=2, dash='dot'))
            fig.add_annotation(x=dc, y=1, yref='paper', text='dresser change', showarrow=False, yanchor='bottom',
                               font=dict(color=ORANGE, size=13))
        if cur is not None and cur['mode'] == mode:
            fig.add_trace(go.Scatter(x=[cur.day], y=[cur.pred], mode='markers', name='current wafer',
                                     marker=dict(size=18, color='rgba(0,0,0,0)', line=dict(color=BLUE, width=3))))
        fig.update_xaxes(range=[max(meta['cut_day'], t1 - window) - 0.05, t1 + 0.15])
    style(fig, height=470, xaxis_title='day', yaxis_title=f'{mode} MRR (nm/min)')
    if not len(d):
        fig.add_annotation(text=f'No {mode} wafers processed yet. Press Play to start.', showarrow=False,
                           x=0.5, y=0.5, xref='paper', yref='paper', font=dict(size=18, color=MUTED))
    return fig


@st.fragment(run_every=0.7 if ss.playing else None)
def live_panel():
    if ss.playing:
        ss.idx = min(N, ss.idx + speed)
        if ss.idx >= N:
            ss.playing = False
    n = ss.idx
    r = res.iloc[:n]
    s = tc.skipped_stats(res, n) if n else None
    cur = r.iloc[-1] if n else None

    k = st.columns(4)
    ok = s is not None and s['n_skip'] > 0
    k[0].metric('Wafers polished', f'{n} / {N}')
    k[1].metric('Skipped metrology',f'{100 * s["n_skip"] / n:.0f}%' if n else '–',
                help='Share of wafers the twin released without a physical measurement')
    k[2].metric('Twin RMSE', f'{s["rmse"]:.2f}' if ok else '–', help='nm/min, on wafers that skipped metrology')
    k[3].metric('Black-box RMSE', f'{s["gbm_rmse"]:.2f}' if ok else '–',
                help='nm/min. Gradient boosting on the same skipped wafers, given the same metrology feedback')
    if n:
        st.caption(f'{s["n_meas"]} wafers sent to metrology · {s["n_skip"]} skipped · '
                   + (f'worst twin error on a skipped wafer {s["max"]:.1f} nm/min (black-box {s["gbm_max"]:.1f}). ' if ok else '')
                   + 'Errors are computed only where the fab relies on the twin.')
    left, right = st.columns([2.3, 1])
    with left:
        st.plotly_chart(stream_chart(r, chart_mode, cur), width='stretch', key='stream')
    with right:
        if cur is None:
            st.info('Press **Play** (or **Next wafer**) in the sidebar to start polishing.')
        else:
            status = ("<div class='status status-meas'>Send to metrology<span>σ above policy limit</span></div>"
                      if cur.measured else
                      "<div class='status status-skip'>Skip metrology<span>Twin is confident</span></div>")
            st.markdown(f"""<div class='wafer-card'>
                <div class='wc-head'><span>Mode {cur['mode']} · day {cur.day:.2f}</span><span>wafer {int(cur.wafer_id)}</span></div>
                <div class='big'>{cur.pred:.1f} <small>± {2 * cur.sigma:.1f} nm/min</small></div>
                <div class='wc-cap'>predicted MRR, 95% band</div>
                {status}
                <div class='kv'><span>σ / policy limit</span><b>{cur.sigma:.2f} / {thr:.1f}</b></div>
                <div class='kv'><span>True MRR</span><b>{cur.MRR_true:.1f}</b></div>
                <div class='kv'><span>Twin error</span><b>{cur.err:+.1f}</b></div>
                <div class='kv'><span>GBM error</span><b>{cur.gbm_err:+.1f}</b></div></div>""", unsafe_allow_html=True)
            st.plotly_chart(waterfall(cur, height=290), width='stretch', key='wf')

    if n:
        log = r.iloc[::-1].head(8)
        tbl = pd.DataFrame({
            'wafer': log.wafer_id.astype(int).astype(str), 'mode': log['mode'], 'day': log.day.round(2),
            'twin MRR': log.pred.round(1), '± 2σ': (2 * log.sigma).round(1), 'true MRR': log.MRR_true.round(1),
            'twin error': log.err.round(1), 'GBM error': log.gbm_err.round(1),
            'decision': np.where(log.measured, 'metrology', 'skip')})
        st.markdown("<div class='section'>Recent wafers</div>", unsafe_allow_html=True)
        st.dataframe(tbl, hide_index=True, width='stretch', height=318)


with tab_live:
    live_panel()

# ----------------------------------------------------------------------------------------------------------------
# Tab 2: physics vs AI decomposition
# ----------------------------------------------------------------------------------------------------------------
with tab_phys:
    n = max(ss.idx, 1)
    r = res.iloc[:n] if ss.idx else res
    share = (r.ai_residual.abs() / r.pred).mean() * 100
    bound_pct = 15
    c = st.columns(4)
    c[0].metric('Average |AI residual|', f'{r.ai_residual.abs().mean():.2f} nm/min')
    c[1].metric('Residual as % of prediction', f'{share:.1f}%')
    c[2].metric('Physics share of prediction', f'{100 - share:.1f}%')
    c[3].metric('Hard bound on AI residual', f'± {bound_pct}%', help='of the mode mean MRR')
    lede('The AI term only corrects what Preston + pad-state physics misses, and it can never exceed its bound.')
    d = r[r['mode'] == chart_mode].tail(60)
    bound = 0.15 * meta['mrr_mean'][meta['MODES'].index(chart_mode)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(len(d)), y=d.ai_residual, marker_color=np.where(d.ai_residual >= 0, BLUE, ORANGE),
                         customdata=np.c_[d.wafer_id, d.physics_term, d.pred],
                         hovertemplate='wafer %{customdata[0]}<br>physics %{customdata[1]:.1f}<br>AI residual %{y:+.2f}'
                                       '<br>twin %{customdata[2]:.1f}<extra></extra>', name='AI residual'))
    for sgn in (1, -1):
        fig.add_hline(y=sgn * bound, line=dict(color=RED, dash='dash'),
                      annotation_text=f'{"+" if sgn > 0 else "−"}bound ({bound:.1f} nm/min)',
                      annotation_position='top right' if sgn > 0 else 'bottom right')
    style(fig, height=380, showlegend=False, xaxis_title=f'last {len(d)} {chart_mode} wafers processed',
          yaxis_title='AI residual (nm/min)')
    fig.update_yaxes(range=[-bound * 1.25, bound * 1.25])
    st.plotly_chart(fig, width='stretch')

    r = r[r['mode'] == chart_mode]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=r.physics_term, y=r.MRR_true, mode='markers', name='physics term only',
                              marker=dict(color=AQUA, size=7, opacity=.7)))
    fig2.add_trace(go.Scatter(x=r.pred, y=r.MRR_true, mode='markers', name='full twin (physics + AI + assimilation)',
                              marker=dict(color=BLUE, size=7, opacity=.7)))
    lo, hi = r.MRR_true.min() - 5, r.MRR_true.max() + 5
    fig2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', line=dict(color=MUTED, dash='dot'), name='perfect'))
    rm_p = np.sqrt(np.mean((r.physics_term - r.MRR_true) ** 2)); rm_t = np.sqrt(np.mean((r.pred - r.MRR_true) ** 2))
    style(fig2, height=430, xaxis_title='predicted MRR (nm/min)', yaxis_title='true MRR (nm/min)',
          title=dict(text=f'{chart_mode} parity: physics term alone RMSE {rm_p:.2f} → full twin RMSE {rm_t:.2f} nm/min', x=0, y=0.98))
    fig2.update_layout(margin=dict(t=80))
    st.plotly_chart(fig2, width='stretch')

# ----------------------------------------------------------------------------------------------------------------
# Tab 3: latent pad state
# ----------------------------------------------------------------------------------------------------------------
with tab_pad:
    lede('The twin carries a latent pad state. Its Preston coefficient k<sub>p</sub> falls as the dresser disk dulls '
         'and the pad glazes, and jumps back when the dresser is changed (orange line). '
         'A black-box model has no memory of this.')
    d = res[res['mode'] == chart_mode]
    now = res.iloc[ss.idx - 1].day if ss.idx else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.day, y=d.kp, mode='lines', line=dict(color=BLUE, width=2.5), name='wear coefficient k_p',
                             hovertemplate='day %{x:.2f}<br>k_p %{y:.1f} nm/min<extra></extra>'))
    for dc in d[d.dresser_change].day:
        fig.add_vline(x=dc, line=dict(color=ORANGE, width=2, dash='dot'))
        fig.add_annotation(x=dc, y=1, yref='paper', yanchor='bottom', text='dresser change', showarrow=False,
                           font=dict(color=ORANGE, size=13))
    if now is not None:
        fig.add_vline(x=now, line=dict(color=INK, width=1.5))
        fig.add_annotation(x=now, y=0, yref='paper', yanchor='bottom', xanchor='left', text=' now', showarrow=False)
    style(fig, height=360, showlegend=False, xaxis_title='day (deployment period)',
          yaxis_title=f'{chart_mode} k_p (nm/min at nominal P, V)')
    fig.update_layout(margin=dict(t=50))
    st.plotly_chart(fig, width='stretch')

    m = meta['MODES'].index(chart_mode)
    rel = d.MRR_true / (d.physics_term / d.z_dresser)          # measured MRR with everything except dresser wear divided out
    bins = pd.cut(d.u_dresser, np.arange(0, 850, 50))
    g = pd.DataFrame({'u': d.u_dresser, 'rel': rel}).groupby(bins, observed=True).agg(
        u=('u', 'mean'), mean=('rel', 'mean'), se=('rel', lambda v: v.std() / np.sqrt(max(len(v), 1))), n=('rel', 'size'))
    xs = np.linspace(0, 800, 100)
    zc = np.mean([(1 - W[f'{k}_amp_d'][m]) + W[f'{k}_amp_d'][m] * np.exp(-W[f'{k}_lam_d'][m] * xs / 800)
                  for k in range(meta['n_ensemble'])], 0)
    r_corr = np.corrcoef(g['mean'], np.interp(g.u, xs, zc))[0, 1]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=g.u, y=g['mean'], mode='markers', name='measured MRR (binned, ± 1 s.e.)',
                              error_y=dict(type='data', array=g.se, color=INK, thickness=1.5),
                              marker=dict(color=INK, size=9), customdata=g.n,
                              hovertemplate='dresser usage %{x:.0f}<br>relative MRR %{y:.3f}<br>%{customdata} wafers<extra></extra>'))
    fig2.add_trace(go.Scatter(x=xs, y=zc, mode='lines', line=dict(color=RED, width=3), name='learned pad-state ODE  z_d(dresser usage)'))
    style(fig2, height=380, xaxis_title='dresser usage since last change', yaxis_title='relative MRR',
          title=dict(text=f'Recovered wear law vs. {len(d)} unseen {chart_mode} wafers:  r = {r_corr:.2f}', x=0, y=0.98))
    fig2.update_layout(margin=dict(t=80))
    st.plotly_chart(fig2, width='stretch')

# ----------------------------------------------------------------------------------------------------------------
# Tab 4: metrology trade-off + savings
# ----------------------------------------------------------------------------------------------------------------
with tab_trade:
    tr = tradeoff_curve()
    cur_skip = 100 * (1 - res_full.measured.mean())
    cur_st = tc.skipped_stats(res_full)
    c = st.columns(4)
    c[0].metric('Wafers skipping metrology', f'{cur_skip:.0f}%')
    c[1].metric('RMSE on skipped wafers', f'{cur_st["rmse"]:.2f} nm/min')
    c[2].metric('95th-pct error on skipped', f'{cur_st["p95"]:.1f} nm/min')
    c[3].metric('GBM RMSE on the same wafers', f'{cur_st["gbm_rmse"]:.2f} nm/min')
    cc = st.columns(2)
    for col, key, lab in [(cc[0], 'rmse', 'RMSE on skipped wafers (nm/min)'),
                          (cc[1], 'max', 'worst |error| on skipped wafers (nm/min)')]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr.skip_pct, y=tr[f'{key}_twin'], mode='lines+markers', name='twin decides (uncertainty-guided)',
                                 line=dict(color=BLUE, width=2.5), marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=tr.skip_pct, y=tr[f'{key}_random'], mode='lines+markers', name='random sampling, same budget',
                                 line=dict(color=MUTED, width=2, dash='dash'), marker=dict(size=8, symbol='square')))
        fig.add_vline(x=cur_skip, line=dict(color=INK, dash='dot'), annotation_text='current policy')
        style(fig, height=380, xaxis_title='% of wafers skipping physical metrology', yaxis_title=lab)
        col.plotly_chart(fig, width='stretch')

    st.markdown("<div class='section'>What this is worth to a fab · illustrative, edit the assumptions</div>",
                unsafe_allow_html=True)
    a = st.columns(4)
    wpm = a[0].number_input('Wafer passes through CMP per month', 1000, 2_000_000, 100_000, 10_000)
    base_rate = a[1].slider('Today: % of wafers measured', 5, 100, 50)
    cost = a[2].number_input('Cost per metrology measurement (USD)', 1.0, 500.0, 25.0, 5.0)
    cyc = a[3].number_input('Metrology minutes per measurement', 0.5, 30.0, 3.0, 0.5)
    saved = wpm * base_rate / 100 * cur_skip / 100
    b = st.columns(3)
    b[0].metric('Measurements avoided per year', f'{12 * saved:,.0f}')
    b[1].metric('Metrology cost avoided per year', f'${12 * saved * cost / 1e6:,.2f} M')
    b[2].metric('Metrology tool-hours freed per year', f'{12 * saved * cyc / 60:,.0f} h')

# ----------------------------------------------------------------------------------------------------------------
# Tab 5: what-if recipe explorer
# ----------------------------------------------------------------------------------------------------------------
with tab_what:
    lede('Recipe exploration without burning test wafers: change pressure, platen speed and consumable age, '
         'and the twin answers instantly (5-model ensemble running live in numpy).')
    l, rr = st.columns([1, 2])
    with l:
        mode = st.radio('Tool mode', meta['MODES'], index=2, horizontal=True, key='wi_mode')
        ref = dict(meta['ref_point'][mode]); rg = meta['ranges'][mode]
        P = st.slider('Chamber pressure P', float(rg['P'][0]) * 0.9, float(rg['P'][1]) * 1.1, float(ref['P']))
        V = st.slider('Platen (stage) rotation V', float(rg['V'][0]) * 0.9, float(rg['V'][1]) * 1.1, float(ref['V']))
        ud = st.slider('Dresser usage since change', 0.0, 800.0, float(ref['u_dresser']))
        ut = st.slider('Pad / table usage since conditioning', 0.0, 400.0, float(ref['u_table']))

    def feats(P, V, ud, ut):
        f = dict(ref)
        f.update(P=P, V=V, PV=P * V, u_dresser=ud, u_table=ut,
                 work=ref['work'] * (P * V) / max(ref['P'] * ref['V'], 1e-9))
        return f

    o = tc.twin_forward(W, meta, mode, feats(P, V, ud, ut))
    base = tc.twin_forward(W, meta, mode, dict(ref))
    with l:
        st.metric('Predicted MRR', f'{o["mu"][0]:.1f} ± {2 * o["sigma"][0]:.1f} nm/min',
                  f'{o["mu"][0] - base["mu"][0]:+.1f} vs typical recipe', delta_color='off')
        st.caption(f'Preston physics {o["phys"][0]:.1f} + AI residual {o["resid"][0]:+.1f} nm/min · '
                   f'wear coefficient k_p = {o["kp"][0]:.1f}')
    with rr:
        xs = np.linspace(0, 800, 120)
        f = feats(P, V, ud, ut); f['u_dresser'] = xs
        cv = tc.twin_forward(W, meta, mode, f)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[cv['mu'] + 2 * cv['sigma'], (cv['mu'] - 2 * cv['sigma'])[::-1]],
                                 fill='toself', fillcolor=BAND, line=dict(width=0), name='± 2σ',
                                 hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=xs, y=cv['phys'], mode='lines', line=dict(color=AQUA, width=2), name='Preston physics'))
        fig.add_trace(go.Scatter(x=xs, y=cv['mu'], mode='lines', line=dict(color=BLUE, width=3), name='twin prediction'))
        fig.add_trace(go.Scatter(x=[ud], y=[o['mu'][0]], mode='markers', marker=dict(size=16, color=BLUE,
                                 line=dict(color='white', width=2)), name='this recipe'))
        style(fig, height=450, xaxis_title='dresser usage since last change', yaxis_title=f'{mode} MRR (nm/min)',
              title=dict(text='Predicted MRR over the dresser life at this recipe', x=0, y=0.98))
        fig.update_layout(margin=dict(t=80))
        st.plotly_chart(fig, width='stretch')

st.markdown("<div class='foot'>Data: 2016 PHM Society Data Challenge (CMP). Twin = Preston backbone with latent pad-state "
            "ODE + bounded residual net (5-member deep ensemble) + Bayesian assimilation of past metrology. "
            "ME6324 · Group 37.</div>", unsafe_allow_html=True)
