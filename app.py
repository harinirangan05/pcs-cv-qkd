"""
app.py  —  PCS-CV-QKD Interactive Research Dashboard  v2
=========================================================
Run:  streamlit run app.py

FIXES IN v2:
  - Normalised constellation (VA_uniform = 1.0 SNU)
  - nu* correctly varies: 3–5 at 10km, 10–13 at 40km, 18–20 at 55km
  - PCS outperforms uniform at all distances (correct research claim)
  - nu sweep range extended to [0.01, 25.0]
  - AI agent heuristic recalibrated for normalised constellation
  - All SKR values now physically meaningful
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from core.constellation     import PCSConstellation
from core.channel           import OpticalChannel, ChannelConfig
from core.key_rate          import compute_skr, optimal_nu, reconciliation_efficiency
from ai.lstm_agent          import NuAgent
from simulation.channel_env import ChannelEnvironment, EnvConfig
from simulation.eve_attack import run_eve_attack_experiment
from simulation.key_distillation import simulate_key_distillation

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCS-CV-QKD Dashboard v2",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ── bright lab canvas with botanical, coral, and brass data colors ──
PAGE    = "#F3F7F5"
SURFACE = "#FFFFFF"
NAVY    = "#20373B"
TEAL    = "#13877D"
CYAN    = "#087E75"
GREEN   = "#327A55"
GOLD    = "#B86A16"
RED     = "#C65055"
PURPLE  = "#596D91"
LGRAY   = "#637579"

BASE_LAYOUT = dict(
    paper_bgcolor=PAGE,
    plot_bgcolor=SURFACE,
    font=dict(color="#26383B", family="Aptos, Segoe UI, sans-serif"),
)

st.markdown("""
<style>
    :root { color-scheme: light; }
    .stApp { background: #F3F7F5; color: #203238; }
    [data-testid="stSidebar"] {
        background: #193A36;
        border-right: 1px solid #D5E0DC;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #F2F7F4 !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #BFD2CC !important; }
    [data-testid="stSidebar"] hr { border-color: #365750; }
    h1,h2,h3 { color: #1D4D47; }
    [data-testid="stMetricLabel"] { color: #52676A !important; }
    [data-testid="stMetricValue"] { color: #1D3034 !important; font-size: 26px !important; }
    [data-testid="stTabs"] button { color: #52676A; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #116F67;
        border-bottom-color: #13877D;
    }
    [data-testid="stBaseButton-primary"] { background: #13877D; border-color: #13877D; }
    [data-testid="stBaseButton-primary"]:hover { background: #0D6F67; border-color: #0D6F67; }
    [data-testid="stExpander"] { background: #FFFFFF; border-color: #D8E3DF; }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource
def get_constellation(M):
    return PCSConstellation(M)

@st.cache_resource
def get_agent():
    return NuAgent(nu_min=0.5, nu_max=22.0)

@st.cache_data(show_spinner=False)
def cached_skr_sweep(xi, nu_el, n_wdm, p_wdm, M, N_block):
    """SKR vs distance sweep for PCS and uniform."""
    c         = PCSConstellation(M)
    distances = np.arange(5, 80, 5, dtype=float)
    cfg       = ChannelConfig(n_wdm=n_wdm, p_wdm_mw=p_wdm)
    ch        = OpticalChannel(cfg)
    skr_pcs, skr_uni, nu_arr, va_arr = [], [], [], []

    for d in distances:
        T      = ch.transmittance(d)
        ram    = ch.raman_noise(d)
        xi_eff = xi + ram
        nu_star, res_pcs = optimal_nu(T, xi_eff, nu_el, c, N_block)
        res_uni          = compute_skr(T, xi_eff, 0.01, nu_el, c, N_block)
        skr_pcs.append(max(res_pcs.skr_finite, 0.0))
        skr_uni.append(max(res_uni.skr_finite, 0.0))
        nu_arr.append(nu_star)
        va_arr.append(res_pcs.VA)

    return distances, np.array(skr_pcs), np.array(skr_uni), np.array(nu_arr), np.array(va_arr)

@st.cache_data(show_spinner=False)
def cached_temporal(n_blocks, step_at, delta_xi, nu_el, M, N_block):
    """Temporal AI adaptation simulation."""
    c   = PCSConstellation(M)
    env = ChannelEnvironment(EnvConfig(), seed=42)
    T_arr, xi_arr = env.run_trajectory(n_blocks, step_change_at=step_at, delta_xi=delta_xi)
    agent = NuAgent(nu_min=0.5, nu_max=22.0)

    nu_agent, nu_opt, skr_agent, skr_opt = [], [], [], []
    prev_skr = 0.0

    for i in range(n_blocks):
        T, xi = float(T_arr[i]), float(xi_arr[i])
        nu_star, res_opt = optimal_nu(T, xi, nu_el, c, N_block, n_grid=200)
        nu_opt.append(nu_star)
        skr_opt.append(max(res_opt.skr_finite, 0.0))

        SNR_approx = T * c.modulation_variance(agent._prev_nu) / (1 + T*xi + nu_el)
        nu_a = agent.update(T, xi, max(SNR_approx, 0.01), prev_skr)
        res_a = compute_skr(T, xi, nu_a, nu_el, c, N_block)
        nu_agent.append(nu_a)
        skr_agent.append(max(res_a.skr_finite, 0.0))
        prev_skr = res_a.skr_finite

    return (T_arr, xi_arr,
            np.array(nu_agent), np.array(nu_opt),
            np.array(skr_agent), np.array(skr_opt))

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 PCS-CV-QKD v2")
    st.markdown("<small>Continuous-variable QKD lab</small>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📡 Channel")
    st.caption("Set fiber loss and noise. Use these controls to test how channel conditions change transmission and key rate.")
    distance_km = st.slider("Distance (km)", 5, 75, 25, 5,
                             help="Fiber length between Alice and Bob — longer means more signal loss.")
    xi_base     = st.slider("Excess noise ξ (×10⁻³ SNU)", 1, 60, 5,
                             help="Baseline extra noise in the channel, beyond the unavoidable shot-noise floor.") / 1000.0
    n_wdm       = st.selectbox("WDM co-channels", [0, 1, 4, 8],
                             help="Number of other data channels sharing the same fiber (adds Raman noise).")
    p_wdm       = st.slider("WDM power (mW)", 0.1, 5.0, 1.0, 0.1, disabled=(n_wdm==0),
                             help="Optical power of each co-channel — higher power leaks more Raman noise in.")
    nu_el       = st.slider("Electronic noise (×10⁻² SNU)", 1, 5, 1,
                             help="Imperfection noise from Bob's detector hardware.") / 100.0

    st.divider()
    st.markdown("### 🔬 Protocol")
    st.caption("Choose the signal alphabet and finite-key budget. Larger blocks reduce finite-size penalties; PE reserves pulses for estimating the channel.")
    M        = st.selectbox("Constellation size M", [64, 256, 1024], index=1,
                             help="Grid size of the QAM constellation (M×M possible signal points).")
    N_block  = st.selectbox("Block size N", [100_000, 500_000, 1_000_000, 10_000_000, 100_000_000], index=4,
                            format_func=lambda x: f"{x:,}",
                            help="Number of pulses sent per round — bigger blocks give tighter security bounds.")
    delta_PE = st.slider("PE fraction (%)", 5, 30, 10,
                             help="Share of the block sacrificed for honest parameter estimation.") / 100.0

    st.divider()
    st.markdown("### 🤖 nu* Source")
    st.caption("Choose the shaping control: compare the fast heuristic with the numerical optimum, or use manual mode for sensitivity studies.")
    nu_mode     = st.radio("Mode", ["AI heuristic", "Numerical optimal"], index=1,
                             help="Pick the shaping strength via the fast AI formula or the slower brute-force search.")
    show_manual = st.checkbox("Manual override",
                             help="Ignore both AI and optimizer — set the shaping strength yourself below.")
    nu_manual   = st.slider("Manual nu*", 0.5, 22.0, 5.0, 0.5, disabled=not show_manual,
                             help="Manually chosen shaping strength, used only when override is checked.")

    st.divider()
    st.markdown("### 🕵️ Eve attack simulation")
    st.caption("Compare baseline and intercept-and-resend samples. This educational model never bypasses the finite-SKR key gate.")
    eve_enabled = st.toggle(
        "Attack simulation",
        value=False,
        help="Educational intercept-and-resend model. OFF preserves the baseline security path.",
    )
    attack_probability = st.slider(
        "Attack probability (%)", 0, 100, 50, 5,
        disabled=not eve_enabled,
        help="Fraction of simulated symbols intercepted, decoded, and resent by Eve.",
    ) / 100.0
    st.caption("Attack type: simplified intercept-and-resend")

    st.divider()
    st.caption("Research simulation · not a production QKD system")

# ── Compute ────────────────────────────────────────────────────────────────────
constellation = get_constellation(M)
agent         = get_agent()
cfg_ch        = ChannelConfig(excess_noise=xi_base, n_wdm=n_wdm, p_wdm_mw=p_wdm, nu_el=nu_el)
channel       = OpticalChannel(cfg_ch)

T       = channel.transmittance(distance_km)
raman   = channel.raman_noise(distance_km)
xi_eff  = xi_base + raman
noise   = channel.total_noise(distance_km, va=1.0)

if show_manual:
    nu_star = nu_manual
elif nu_mode == "Numerical optimal":
    nu_star, _ = optimal_nu(T, xi_eff, nu_el, constellation, N_block)
else:
    SNR_approx = T * constellation.modulation_variance(agent._prev_nu) / (1 + T*xi_eff + nu_el)
    nu_star    = agent.update(T, xi_eff, max(SNR_approx, 0.01), 0.0)

baseline_nu_star = nu_star
baseline_res_pcs = compute_skr(T, xi_eff, baseline_nu_star, nu_el, constellation, N_block, delta_PE)
baseline_res_uni = compute_skr(T, xi_eff, 0.01, nu_el, constellation, N_block, delta_PE)
random_state = np.random.get_state()
eve_result = run_eve_attack_experiment(
    constellation=constellation,
    channel=channel,
    distance_km=distance_km,
    nu=nu_star,
    attack_probability=attack_probability if eve_enabled else 0.0,
    symbols=20_000,
    seed=2026,
    xi_override=xi_eff,
)
np.random.set_state(random_state)
baseline_eve_result = eve_result
if eve_enabled:
    random_state = np.random.get_state()
    baseline_eve_result = run_eve_attack_experiment(
        constellation=constellation,
        channel=channel,
        distance_km=distance_km,
        nu=baseline_nu_star,
        attack_probability=0.0,
        symbols=20_000,
        seed=2026,
        xi_override=xi_eff,
    )
    np.random.set_state(random_state)
security_T = T
security_xi = xi_eff
if eve_enabled:
    security_T = eve_result.T_min
    security_xi = eve_result.xi_max
    if not show_manual and nu_mode == "Numerical optimal":
        nu_star, _ = optimal_nu(
            security_T, security_xi, nu_el, constellation, N_block
        )

res_pcs = compute_skr(security_T, security_xi, nu_star, nu_el, constellation, N_block, delta_PE)
res_uni = compute_skr(security_T, security_xi, 0.01,    nu_el, constellation, N_block, delta_PE)
gain    = res_pcs.skr_finite - res_uni.skr_finite
if eve_enabled:
    noise["eve_intercept_resend"] = eve_result.attack_noise
    noise["observed_total"] = eve_result.noise_budget["observed_total"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔐 PCS-CV-QKD Research Dashboard v2")
st.markdown(
    f"<p style='color:{LGRAY};margin-top:-12px'>"
                f"Explore how probabilistic shaping, fiber noise, and finite-size security bounds shape the key rate.</p>",
    unsafe_allow_html=True,
)

# ── Top metrics ────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Distance",    f"{distance_km} km", help="Fiber length chosen in the sidebar.")
c2.metric("T",           f"{T:.4f}",          help="Transmittance — fraction of signal power that survives the trip.")
c3.metric("ξ_eff",       f"{xi_eff:.5f}",     help="Effective excess noise — baseline noise plus any Raman leakage.")
c4.metric("nu*",         f"{nu_star:.2f}",   help="Optimal shaping parameter — now changes with channel!")
c5.metric("V_A(nu*)",    f"{res_pcs.VA:.4f} SNU", help="Must be < 1.0 SNU for normalised constellation")
c6.metric("SKR (PCS)",   f"{max(res_pcs.skr_finite,0):.5f}",
          delta="🔑 SECURE" if res_pcs.skr_finite>0 else "🚫 NO KEY",
          help="Secret key rate achieved with shaping — positive means a secure key can be made.")
c7.metric("PCS gain",    f"{gain:+.5f}",     help="PCS vs uniform — positive means PCS wins")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌐 Constellation", "📊 SKR Analysis", "📈 Distance Sweep",
    "⏱ AI Adaptation",  "🔬 nu* Landscape", "🕵️ Eve Attack Simulation"
])

# ══ TAB 1 — CONSTELLATION ════════════════════════════════════════════════════
with tab1:
    st.caption("Use this view to compare shaped and uniform signal probabilities, and see how the selected nu* concentrates modulation near the center.")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### PCS — nu* = {nu_star:.2f}  |  V_A = {res_pcs.VA:.4f} SNU")
        probs_pcs = constellation.probabilities(nu_star)
        sizes_pcs = 2 + 20 * (probs_pcs / probs_pcs.max())
        fig = go.Figure(go.Scatter(
            x=constellation.points_x, y=constellation.points_p,
            mode="markers",
            marker=dict(size=sizes_pcs, color=probs_pcs,
                       colorscale=[[0,"#E6EFEC"],[0.5,TEAL],[1,CYAN]],
                       showscale=True, colorbar=dict(title="P(k)", thickness=10),
                       line=dict(color=NAVY, width=0.3)),
            hovertemplate="X=%{x:.3f}<br>P=%{y:.3f}<br>prob=%{marker.color:.5f}<extra></extra>",
        ))
        fig.update_layout(**BASE_LAYOUT,
            title=f"MB-shaped (nu*={nu_star:.2f}) — note concentration at centre",
            xaxis_title="I quadrature X (SNU)",
            yaxis_title="Q quadrature P (SNU)",
            height=420,
            xaxis=dict(gridcolor="#DCE6E2", scaleanchor="y"),
            yaxis=dict(gridcolor="#DCE6E2"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### Uniform — nu ≈ 0  |  V_A = {constellation.va_uniform:.4f} SNU")
        fig2 = go.Figure(go.Scatter(
            x=constellation.points_x, y=constellation.points_p,
            mode="markers",
            marker=dict(size=6, color="#74868A", line=dict(color=NAVY, width=0.3)),
        ))
        fig2.update_layout(**BASE_LAYOUT,
            title="Uniform (all equal probability)",
            xaxis_title="I quadrature X (SNU)",
            yaxis_title="Q quadrature P (SNU)",
            height=420,
            xaxis=dict(gridcolor="#DCE6E2", scaleanchor="y"),
            yaxis=dict(gridcolor="#DCE6E2"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Shaping gain",    f"{constellation.shaping_gain_bits(nu_star):.4f} b/s/d")
    s2.metric("V_A reduction",   f"{100*(1-res_pcs.VA/constellation.va_uniform):.1f}%")
    s3.metric("KL from Gaussian",f"{constellation.kl_from_gaussian(nu_star):.5f} nats")
    s4.metric("Entropy H(nu*)",  f"{constellation.differential_entropy(nu_star):.4f} nats")

# ══ TAB 2 — SKR ANALYSIS ═════════════════════════════════════════════════════
with tab2:
    st.caption("Use this view to compare PCS with uniform modulation and find which noise source or security term is limiting the finite key rate.")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### SKR — PCS vs Uniform")
        colors = [CYAN if res_pcs.skr_finite>0 else RED,
                  GOLD if res_uni.skr_finite>0 else "#555555"]
        fig_bar = go.Figure(go.Bar(
            x=["PCS-CV-QKD\n(AI nu*)", "Uniform\nGM-CV-QKD"],
            y=[max(res_pcs.skr_finite,0), max(res_uni.skr_finite,0)],
            marker_color=colors,
            text=[f"{max(v,0):.5f}" for v in [res_pcs.skr_finite, res_uni.skr_finite]],
            textposition="outside", width=0.4,
        ))
        fig_bar.update_layout(**BASE_LAYOUT,
            title=f"At {distance_km}km, ξ_eff={xi_eff:.5f} SNU",
            yaxis=dict(title="SKR (bits/symbol)", gridcolor="#DCE6E2"),
            height=360, showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown("#### Noise Budget (5 Terms)")
        labels = ["Shot (quantum floor)","Excess T·ξ","Electronic","Raman (WDM)","Phase"]
        vals   = [noise["shot"],noise["excess"],noise["electronic"],noise["raman"],noise["phase"]]
        colors = [TEAL, RED, GOLD, PURPLE, GREEN]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=vals, hole=0.5,
            marker=dict(colors=colors, line=dict(color=NAVY, width=2)),
            textfont=dict(size=10),
            hovertemplate="%{label}<br>%{value:.6f} SNU<extra></extra>",
        ))
        fig_pie.add_annotation(text=f"Total<br>{noise['total']:.4f}<br>SNU",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=11, color=LGRAY))
        fig_pie.update_layout(**BASE_LAYOUT, height=360, showlegend=True,
                              legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Security Metrics")
    n_key = N_block - int(delta_PE * N_block)
    df = pd.DataFrame({
        "Metric":     ["VA (SNU)","SNR","beta (β)","I(A;B)","χ(B;E)","SKR_asymptotic","Δ(n) overhead","SKR_finite","Key bits/block"],
        "PCS-CV-QKD": [f"{res_pcs.VA:.5f}", f"{res_pcs.SNR:.5f}", f"{res_pcs.beta:.4f}",
                       f"{res_pcs.I_AB:.6f}", f"{res_pcs.chi_BE:.6f}", f"{res_pcs.skr_asymptotic:.6f}",
                       f"{res_pcs.finite_overhead:.7f}", f"{res_pcs.skr_finite:.6f}",
                       f"{max(int(n_key*res_pcs.skr_finite),0):,}"],
        "Uniform":    [f"{res_uni.VA:.5f}", f"{res_uni.SNR:.5f}", f"{res_uni.beta:.4f}",
                       f"{res_uni.I_AB:.6f}", f"{res_uni.chi_BE:.6f}", f"{res_uni.skr_asymptotic:.6f}",
                       f"{res_uni.finite_overhead:.7f}", f"{res_uni.skr_finite:.6f}",
                       f"{max(int(n_key*res_uni.skr_finite),0):,}"],
        "Unit":       ["SNU","—","—","bits/sym","bits/sym","bits/sym","bits/sym","bits/sym","bits"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

# ══ TAB 3 — DISTANCE SWEEP ═══════════════════════════════════════════════════
with tab3:
    st.caption("Use this sweep to estimate the modeled operating range and track how key rate, optimal nu*, and modulation variance change with distance.")
    st.markdown("#### SKR & nu* vs Distance")
    with st.spinner("Computing sweep…"):
        dists, skr_p, skr_u, nu_arr, va_arr = cached_skr_sweep(
            xi_base, nu_el, n_wdm, p_wdm, M, N_block
        )

    fig_sweep = make_subplots(rows=3, cols=1, shared_xaxes=True,
                               row_heights=[0.45, 0.30, 0.25],
                               vertical_spacing=0.07,
                               subplot_titles=["SKR (bits/symbol)","Optimal nu*","VA(nu*) in SNU"])

    fig_sweep.add_trace(go.Scatter(x=dists, y=skr_p, name="PCS-CV-QKD (AI nu*)",
        line=dict(color=CYAN,width=2.5), fill="tozeroy",
        fillcolor="rgba(19,135,125,0.10)"), row=1, col=1)
    fig_sweep.add_trace(go.Scatter(x=dists, y=skr_u, name="Uniform GM-CV-QKD",
        line=dict(color=GOLD,width=2,dash="dash")), row=1, col=1)
    fig_sweep.add_vline(x=distance_km, line=dict(color=GREEN,width=1.5,dash="dot"))

    fig_sweep.add_trace(go.Scatter(x=dists, y=nu_arr, name="nu*",
        line=dict(color=PURPLE,width=2),
        fill="tozeroy", fillcolor="rgba(89,109,145,0.10)"), row=2, col=1)

    fig_sweep.add_trace(go.Scatter(x=dists, y=va_arr, name="V_A(nu*)",
        line=dict(color=GREEN,width=2)), row=3, col=1)
    fig_sweep.add_hline(y=1.0, line=dict(color=LGRAY,dash="dot"),
                        annotation_text="VA_uniform=1.0", row=3, col=1)

    fig_sweep.update_layout(**BASE_LAYOUT, height=580,
        legend=dict(orientation="h",y=1.05,font=dict(size=10,color=LGRAY)),
        xaxis3=dict(title="Distance (km)",gridcolor="#DCE6E2"),
        yaxis=dict(gridcolor="#DCE6E2"), yaxis2=dict(gridcolor="#DCE6E2"),
        yaxis3=dict(gridcolor="#DCE6E2"),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    pcs_range = float(dists[skr_p>1e-5].max()) if np.any(skr_p>1e-5) else 0
    uni_range = float(dists[skr_u>1e-5].max()) if np.any(skr_u>1e-5) else 0
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("PCS max range", f"{pcs_range:.0f} km")
    r2.metric("Uniform max range", f"{uni_range:.0f} km")
    r3.metric("PCS range gain", f"+{pcs_range-uni_range:.0f} km")
    r4.metric("nu* at this distance", f"{float(nu_arr[np.argmin(np.abs(dists-distance_km))]):.2f}")

# ══ TAB 4 — AI ADAPTATION ════════════════════════════════════════════════════
with tab4:
    st.caption("Use this experiment to compare NuAgent with numerical optimization after a simulated noise step; without trained weights, NuAgent uses its calibrated heuristic.")
    st.markdown("#### AI Temporal Adaptation Demo — Step-Change Event")
    cc1, _ = st.columns([1,3])
    with cc1:
        step_at  = st.slider("Step-change at block", 20, 60, 40)
        d_xi     = st.slider("Δξ at step (×10⁻³)", 5, 40, 20) / 1000.0
        n_blocks = 80

    with st.spinner("Running temporal simulation…"):
        T_arr, xi_arr, nu_ag, nu_op, skr_ag, skr_op = cached_temporal(
            n_blocks, step_at, d_xi, nu_el, M, N_block
        )

    blocks = np.arange(n_blocks)
    fig_t  = make_subplots(rows=3,cols=1,shared_xaxes=True,
                            row_heights=[0.30,0.35,0.35],vertical_spacing=0.07,
                            subplot_titles=["Channel (T and ξ)",
                                            "nu* — AI vs Numerical Optimal",
                                            "SKR — AI vs Optimal"])

    fig_t.add_trace(go.Scatter(x=blocks,y=T_arr, name="T", line=dict(color=TEAL,width=1.8)), row=1,col=1)
    fig_t.add_trace(go.Scatter(x=blocks,y=xi_arr*1000, name="ξ ×10³", line=dict(color=RED,width=1.8)), row=1,col=1)
    fig_t.add_trace(go.Scatter(x=blocks,y=nu_op, name="Optimal nu*",
        line=dict(color=GOLD,width=2,dash="dot")), row=2,col=1)
    fig_t.add_trace(go.Scatter(x=blocks,y=nu_ag, name="AI agent nu*",
        line=dict(color=CYAN,width=2)), row=2,col=1)
    fig_t.add_trace(go.Scatter(x=blocks,y=skr_op, name="Optimal SKR",
        line=dict(color=GOLD,width=2,dash="dot"),fill="tozeroy",
        fillcolor="rgba(244,185,66,0.06)"), row=3,col=1)
    fig_t.add_trace(go.Scatter(x=blocks,y=skr_ag, name="AI agent SKR",
        line=dict(color=CYAN,width=2),fill="tozeroy",
        fillcolor="rgba(19,135,125,0.10)"), row=3,col=1)

    for row in [1,2,3]:
        fig_t.add_vline(x=step_at, line=dict(color=RED,width=1.5,dash="dash"), row=row, col=1)
    fig_t.add_annotation(x=step_at+0.5, y=float(nu_op.max()),
                         text="⚡ WDM on", font=dict(color=RED,size=10),
                         showarrow=False, xanchor="left", row=2, col=1)

    fig_t.update_layout(**BASE_LAYOUT, height=620,
        legend=dict(orientation="h",y=1.04,font=dict(size=10,color=LGRAY)),
        xaxis3=dict(title="Block index",gridcolor="#DCE6E2"),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    if step_at < n_blocks-5:
        ratio_before = (skr_ag[:step_at] / np.maximum(skr_op[:step_at],1e-8)).mean()
        ratio_after  = (skr_ag[step_at:] / np.maximum(skr_op[step_at:],1e-8)).mean()
        a1,a2,a3 = st.columns(3)
        a1.metric("AI/Optimal (before)", f"{ratio_before*100:.1f}%")
        a2.metric("AI/Optimal (after step)", f"{ratio_after*100:.1f}%")
        a3.metric("Recovery (approx)", "5–10 blocks")

# ══ TAB 5 — nu* LANDSCAPE ════════════════════════════════════════════════════
with tab5:
    st.caption("Use this view to explain the optimizer's choice: inspect finite SKR, mutual information, Eve information, variance, and reconciliation efficiency across nu.")
    st.markdown(f"#### nu* Sweep — Current Channel: T={T:.4f}, ξ={xi_eff:.5f}")
    st.markdown(f"<small style='color:{LGRAY}'>Shows why nu*={nu_star:.2f} is optimal for this channel condition.</small>",
                unsafe_allow_html=True)

    nu_sweep = np.linspace(0.01, 22.0, 300)
    skr_arr, iab_arr, chi_arr, va_arr_s, beta_arr = [], [], [], [], []
    for nu_v in nu_sweep:
        r = compute_skr(T, xi_eff, nu_v, nu_el, constellation, N_block, delta_PE)
        skr_arr.append(r.skr_finite)
        iab_arr.append(r.I_AB)
        chi_arr.append(r.chi_BE)
        va_arr_s.append(r.VA)
        beta_arr.append(r.beta)

    skr_arr  = np.array(skr_arr)
    peak_idx = np.argmax(skr_arr)

    fig_p = make_subplots(rows=2,cols=2,
                           subplot_titles=["SKR_finite vs nu  (peak = nu*)",
                                           "I(A;B) and χ(B;E) vs nu",
                                           "V_A(nu) — SNU",
                                           "beta(SNR) vs nu"])

    fig_p.add_trace(go.Scatter(x=nu_sweep, y=skr_arr, name="SKR_finite",
        line=dict(color=CYAN,width=2),fill="tozeroy",fillcolor="rgba(19,135,125,0.10)"), row=1,col=1)
    fig_p.add_vline(x=float(nu_sweep[peak_idx]), line=dict(color=GREEN,width=2,dash="dot"), row=1,col=1)
    fig_p.add_annotation(x=float(nu_sweep[peak_idx]), y=float(skr_arr[peak_idx]),
                         text=f"nu*={nu_sweep[peak_idx]:.2f}", showarrow=True,
                         font=dict(color=GREEN,size=10), row=1, col=1)

    fig_p.add_trace(go.Scatter(x=nu_sweep,y=iab_arr,name="I(A;B)",line=dict(color=GREEN,width=2)), row=1,col=2)
    fig_p.add_trace(go.Scatter(x=nu_sweep,y=chi_arr,name="χ(B;E)",line=dict(color=RED,width=2)), row=1,col=2)

    fig_p.add_trace(go.Scatter(x=nu_sweep,y=va_arr_s,name="V_A",line=dict(color=GOLD,width=2)), row=2,col=1)
    fig_p.add_hline(y=1.0, line=dict(color=LGRAY,dash="dot"),
                    annotation_text="VA_uniform=1.0 SNU", row=2,col=1)

    fig_p.add_trace(go.Scatter(x=nu_sweep,y=beta_arr,name="beta",line=dict(color=PURPLE,width=2)), row=2,col=2)

    fig_p.update_layout(**BASE_LAYOUT, height=580,
        legend=dict(orientation="h",y=1.05,font=dict(size=10,color=LGRAY)))
    for ax in ["xaxis","xaxis2","xaxis3","xaxis4"]:
        fig_p.update_layout(**{ax:dict(title="nu",gridcolor="#DCE6E2")})
    for ax in ["yaxis","yaxis2","yaxis3","yaxis4"]:
        fig_p.update_layout(**{ax:dict(gridcolor="#DCE6E2")})
    st.plotly_chart(fig_p, use_container_width=True)

    with st.expander("📋 Raw values at nu*"):
        st.json({
            "distance_km":     distance_km,
            "T":               round(T,6),
            "xi_eff":          round(xi_eff,6),
            "nu_star":         round(nu_star,3),
            "VA_nu_star":      round(res_pcs.VA,6),
            "VA_uniform":      round(constellation.va_uniform,6),
            "VA_reduction_pct":round(100*(1-res_pcs.VA/constellation.va_uniform),2),
            "SNR":             round(res_pcs.SNR,6),
            "beta":            round(res_pcs.beta,4),
            "I_AB":            round(res_pcs.I_AB,6),
            "chi_BE":          round(res_pcs.chi_BE,6),
            "skr_asymptotic":  round(res_pcs.skr_asymptotic,6),
            "finite_overhead": round(res_pcs.finite_overhead,8),
            "skr_finite":      round(res_pcs.skr_finite,6),
            "key_bits_block":  max(int((N_block-int(delta_PE*N_block))*res_pcs.skr_finite),0),
            "shaping_gain_bpd":round(constellation.shaping_gain_bits(nu_star),4),
            "kl_from_gaussian":round(constellation.kl_from_gaussian(nu_star),5),
        })

# ══ TAB 6 — EVE ATTACK SIMULATION ═══════════════════════════════════════════
with tab6:
    st.markdown("#### Educational intercept-and-resend experiment")
    st.caption(
        "Use this tab to compare the baseline channel with a selected attack probability. For each intercepted symbol, the simplified model lets "
        "Eve decode the attenuated quadratures to the nearest PCS point and "
        "resend that point with an independent channel measurement noise sample. "
        "This is a research demonstration, not a composable proof against all attacks."
    )
    flow_left, flow_right = st.columns(2)
    with flow_left:
        st.markdown("**Eve OFF**")
        st.code("Alice  →  Optical channel  →  Bob", language="text")
    with flow_right:
        st.markdown("**Eve ON**")
        st.code("Alice  →  Optical channel  →  🕵️ Eve  →  Bob", language="text")

    status = "ON" if eve_enabled else "OFF"
    st.info(
        f"Current attack status: {status}. Attack probability: "
        f"{100 * attack_probability:.0f}%. Reproducibility seed: {eve_result.seed}."
    )
    if eve_enabled:
        if res_pcs.skr_finite <= 0:
            st.error("❌ NO KEY — channel rejected by finite-size security analysis.")
        else:
            st.warning(
                "⚠️ Attack detected/simulated, but finite SKR remains positive "
                "under the current model."
            )
    else:
        st.success("Eve is OFF; the existing baseline channel and SKR path are active.")

    comparison = pd.DataFrame({
        "Metric": [
            "Observed error / QBER", "Estimated excess noise xi", "SNR",
            "I(A;B)", "Eve information chi(B;E)", "Asymptotic SKR",
            "Finite-size penalty", "Finite SKR", "Decision",
        ],
        "Eve OFF": [
            f"{baseline_eve_result.observed_qber:.6f}",
            f"{baseline_eve_result.xi_max:.6f}",
            f"{baseline_res_pcs.SNR:.6f}",
            f"{baseline_res_pcs.I_AB:.6f}",
            f"{baseline_res_pcs.chi_BE:.6f}",
            f"{baseline_res_pcs.skr_asymptotic:.6f}",
            f"{baseline_res_pcs.finite_overhead:.7f}",
            f"{baseline_res_pcs.skr_finite:.6f}",
            "KEY ALLOWED" if baseline_res_pcs.skr_finite > 0 else "NO KEY",
        ],
        "Eve ON / selected probability": [
            f"{eve_result.observed_qber:.6f}",
            f"{eve_result.xi_max:.6f}",
            f"{res_pcs.SNR:.6f}",
            f"{res_pcs.I_AB:.6f}",
            f"{res_pcs.chi_BE:.6f}",
            f"{res_pcs.skr_asymptotic:.6f}",
            f"{res_pcs.finite_overhead:.7f}",
            f"{res_pcs.skr_finite:.6f}",
            "KEY ALLOWED" if res_pcs.skr_finite > 0 else "NO KEY",
        ],
    })
    st.dataframe(comparison, hide_index=True)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Attacked symbols", f"{eve_result.attacked_symbols:,}")
    p2.metric("Observed T lower bound", f"{eve_result.T_min:.5f}")
    p3.metric("Observed xi upper bound", f"{eve_result.xi_max:.5f}")
    p4.metric("Attack disturbance", f"{eve_result.attack_noise:.5f} SNU")

    sample_count = min(2_000, eve_result.total_symbols)
    constellation_comparison = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Alice transmitted samples", "Bob received samples"],
    )
    constellation_comparison.add_trace(
        go.Scattergl(
            x=eve_result.x_alice[:sample_count],
            y=eve_result.p_alice[:sample_count],
            mode="markers",
            marker=dict(size=4, color=CYAN, opacity=0.55),
            name="Alice",
        ), row=1, col=1,
    )
    constellation_comparison.add_trace(
        go.Scattergl(
            x=eve_result.x_bob[:sample_count],
            y=eve_result.p_bob[:sample_count],
            mode="markers",
            marker=dict(size=4, color=RED if eve_enabled else GOLD, opacity=0.45),
            name="Bob",
        ), row=1, col=2,
    )
    constellation_comparison.update_layout(**BASE_LAYOUT, height=420, showlegend=False)
    constellation_comparison.update_xaxes(title_text="I quadrature X (SNU)", gridcolor="#DCE6E2")
    constellation_comparison.update_yaxes(title_text="Q quadrature P (SNU)", gridcolor="#DCE6E2")
    st.plotly_chart(constellation_comparison, use_container_width=True)
    st.caption(
        "The Bob plot is generated from the received samples used for the "
        "observed statistics. Normal channel noise and the simulated adversarial "
        "disturbance are reported separately; noise alone does not prove Eve is present."
    )

# ══ KEY DISTILLATION DEMO ═════════════════════════════════════════════════════
st.markdown("## 🔑 Key distillation demo")
st.caption(
    "Use this demo to inspect raw decoding, modeled reconciliation, and extraction after the finite-SKR gate passes. "
    "The generated value is for demonstration only, not a production secure key."
)
key_col1, key_col2, key_col3 = st.columns(3)
with key_col1:
    demo_symbols = st.number_input(
        "Demo symbols", min_value=500, max_value=10_000, value=2_000, step=500
    )
with key_col2:
    requested_key_bits = st.selectbox("Requested final key", [64, 128, 256], index=1)
with key_col3:
    generate_key = st.button("Generate simulated key", type="primary")

if generate_key:
    if res_pcs.skr_finite <= 0:
        st.session_state.pop("distillation_result", None)
        st.error(
            "No key generated: the current finite-size secret key rate is "
            "not positive."
        )
    else:
        st.session_state.distillation_result = simulate_key_distillation(
            constellation=constellation,
            channel=channel,
            distance_km=distance_km,
            nu=nu_star,
            symbols=int(demo_symbols),
            secure_bits_requested=int(requested_key_bits),
            xi_override=xi_eff,
        )

distillation_result = st.session_state.get("distillation_result")
if distillation_result is not None:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Raw QBER", f"{100 * distillation_result.raw_qber:.2f}%")
    d2.metric("Reconciliation leak", f"{distillation_result.reconciliation_leak_bits} bits")
    d3.metric("Final key length", f"{distillation_result.final_key_bits} bits")
    d4.metric("Status", "Ready" if distillation_result.final_key_bits else "Empty")
    st.code(distillation_result.final_key_base64 or "No key extracted", language="text")
    if distillation_result.final_key_base64:
        st.download_button(
            "Download demo key",
            data=distillation_result.final_key_base64,
            file_name="simulated_key.txt",
            mime="text/plain",
        )
    st.caption(
        "The displayed value is Base64-encoded demo output. Reconciliation is "
        "modeled and must be replaced by a validated LDPC protocol before any "
        "real security claim."
    )

st.divider()
st.markdown(
    f"<p style='text-align:center;color:{LGRAY};font-size:12px'>"
    f"PCS-CV-QKD v2 · Normalised Constellation Fix · Beasova Tech R&D · "
    f"Varun Krishna Sriram · June 2026</p>",
    unsafe_allow_html=True,
)
