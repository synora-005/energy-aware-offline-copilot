"""Streamlit dashboard for the Energy-Aware Offline Copilot (UI v4).

  streamlit run dashboard.py

Same data and logic as before; presentation only, plus a live power panel
and a sidebar chat that answers from computed facts.
"""
import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

import dash_data as d
from explainer import MODEL, explain_all, chat_answer
from facts import build
from save_insights import save

st.set_page_config(page_title="Energy-Aware Offline Copilot", layout="wide",
                   page_icon="🌿")

CSS = """
<style>
.block-container {padding-top: 4.5rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {display:flex; align-items:center; gap:14px; margin-bottom:2px;}
.hero .logo {width:44px; height:44px; border-radius:12px; background:#16a34a22;
  border:1px solid #22c55e66; display:flex; align-items:center; justify-content:center;
  font-size:24px;}
.hero .t {font-size:1.9rem; font-weight:700; line-height:1.1;}
.hero .s {color:#94a3b8; font-size:0.9rem;}
.kpi {background:#111a2e; border:1px solid #1f2b45; border-radius:14px;
  padding:16px 18px; height:100%;}
.kpi-l {color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:.06em;}
.kpi-v {font-size:1.9rem; font-weight:700; margin-top:4px;}
.kpi-s {color:#94a3b8; font-size:0.8rem; margin-top:2px;}
.green {color:#22c55e;} .amber {color:#f59e0b;} .red {color:#ef4444;} .blue {color:#38bdf8;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem;
  font-weight:600; border:1px solid; margin-right:6px; vertical-align:middle;}
.pill.g {color:#22c55e; border-color:#22c55e66; background:#22c55e18;}
.pill.a {color:#f59e0b; border-color:#f59e0b66; background:#f59e0b18;}
.pill.b {color:#38bdf8; border-color:#38bdf866; background:#38bdf818;}
.pill.x {color:#94a3b8; border-color:#475569; background:#47556918;}
.action {background:linear-gradient(90deg,#16a34a22,#111a2e); border:1px solid #22c55e55;
  border-radius:14px; padding:16px 20px; margin:6px 0 14px 0;}
.action .h {font-weight:700; color:#22c55e; margin-bottom:4px;}
div[data-testid="stMetric"] {background:#111a2e; border:1px solid #1f2b45;
  border-radius:14px; padding:14px 16px;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
button[data-baseweb="tab"] {font-weight:600;}

/* ---------- glass theme ---------- */
.stApp {
  background:
    radial-gradient(700px 420px at 8% 4%,  rgba(34,197,94,0.22), transparent 60%),
    radial-gradient(640px 420px at 92% 12%, rgba(56,189,248,0.18), transparent 60%),
    radial-gradient(760px 520px at 55% 100%, rgba(139,92,246,0.14), transparent 60%),
    #0b1220;
  background-attachment: fixed;
}
.kpi, div[data-testid="stMetric"],
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(145deg, rgba(255,255,255,0.09), rgba(255,255,255,0.025));
  backdrop-filter: blur(14px) saturate(140%);
  -webkit-backdrop-filter: blur(14px) saturate(140%);
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 8px 28px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.10);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
.kpi:hover, div[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  border-color: rgba(34,197,94,0.45);
  box-shadow: 0 12px 34px rgba(0,0,0,0.42), 0 0 0 1px rgba(34,197,94,0.15);
}
.action {
  background: linear-gradient(120deg, rgba(34,197,94,0.20), rgba(255,255,255,0.04));
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(34,197,94,0.40);
  box-shadow: 0 8px 28px rgba(0,0,0,0.35);
}
section[data-testid="stSidebar"] {
  background: rgba(11,18,32,0.72);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border-right: 1px solid rgba(255,255,255,0.08);
}
header[data-testid="stHeader"] {
  background: rgba(11,18,32,0.55);
  backdrop-filter: blur(10px);
}
button[data-baseweb="tab"][aria-selected="true"] {text-shadow: 0 0 14px rgba(34,197,94,0.55);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

GREY, GREEN, BLUE = "#64748b", "#22c55e", "#38bdf8"


def kpi(label, value, sub="", cls="", icon="", badge=None):
    b = (f'<span class="pill {badge[1]}" style="float:right;margin:0">'
         f'{badge[0]}</span>') if badge else ""
    return (f'<div class="kpi">{b}<div class="kpi-l">{icon} {label}</div>'
            f'<div class="kpi-v {cls}">{value}</div>'
            f'<div class="kpi-s">{sub}</div></div>')


def pill(text, cls="x"):
    return f'<span class="pill {cls}">{text}</span>'


def run_chart(piv):
    """Grouped bars per seed; y-axis always starts at 0."""
    long = piv.reset_index().melt("seed", var_name="variant", value_name="Wh")
    return (alt.Chart(long).mark_bar().encode(
        x=alt.X("seed:N", title="seed (same seed = identical work)"),
        xOffset="variant:N",
        y=alt.Y("Wh:Q", title="Wh per window", scale=alt.Scale(zero=True)),
        color=alt.Color("variant:N", title=None,
                        scale=alt.Scale(domain=["baseline", "optimized"],
                                        range=[GREY, GREEN])),
        tooltip=["seed", "variant", alt.Tooltip("Wh:Q", format=".3f")])
        .properties(height=260))


def ci_chart(scenarios):
    """Saving per scenario with its 95% confidence interval."""
    df = pd.DataFrame([{"scenario": s["scenario"].capitalize(),
                        "saving": s["saving_wh"],
                        "lo": s["ci95_wh"][0], "hi": s["ci95_wh"][1]}
                       for s in scenarios])
    base = alt.Chart(df).encode(y=alt.Y("scenario:N", title=None))
    bars = base.mark_bar(color=GREEN, opacity=0.55, size=28).encode(
        x=alt.X("saving:Q", title="Saving (Wh per window), line = 95% CI",
                scale=alt.Scale(zero=True)),
        tooltip=["scenario", alt.Tooltip("saving:Q", format=".3f"),
                 alt.Tooltip("lo:Q", format=".3f"),
                 alt.Tooltip("hi:Q", format=".3f")])
    rule = base.mark_rule(color="#e5e7eb", strokeWidth=3).encode(x="lo:Q", x2="hi:Q")
    return (bars + rule).properties(height=150)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🌿 Energy Copilot")
    st.markdown(pill("MEASURED", "g") + " battery capacity drop, real hardware",
                unsafe_allow_html=True)
    st.markdown(pill("SYNTHETIC", "a") + " background app (simulated workload)",
                unsafe_allow_html=True)
    st.markdown(pill("MODEL-BASED", "b") + " predictions, not yet verified",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("### 💬 Ask the copilot")
    st.caption("Answers come from computed facts only; every number is "
               "verified before you see it.")
    chat_hist = st.session_state.setdefault("chat", [])
    box = st.container(height=340)
    with box:
        if not chat_hist:
            st.caption("Try: How much energy did batching save?")
        for m in chat_hist:
            with st.chat_message(m["role"]):
                st.write(m["text"])
                if m.get("source"):
                    st.caption("SLM, numbers verified" if m["source"] == "slm"
                               else "fallback: written by code from measured data")
    q = st.chat_input("Ask about your results")
    if q:
        chat_hist.append({"role": "user", "text": q})
        with st.spinner("Local model is thinking..."):
            ans = chat_answer(q, build())
        chat_hist.append({"role": "assistant", "text": ans["text"],
                          "source": ans["source"]})
        st.rerun()

# ---------------------------------------------------------------- header
st.markdown(
    '<div class="hero"><div class="logo">🌿</div><div>'
    '<div class="t">Energy-Aware Offline Copilot</div>'
    f'<div class="s">Local SLM: {MODEL} (about 1.5B parameters, Q4_K_M, offline via '
    'Ollama) &nbsp;|&nbsp; Storage: local SQLite &nbsp;|&nbsp; Sync: durable outbox '
    'with idempotency keys</div></div></div>', unsafe_allow_html=True)
st.write("")


# ------------------------------------------------------------ status strip
def status_strip():
    online = d.online()
    q = d.queue_counts()
    c = st.columns(5)
    c[0].markdown(kpi("Cloud link", "ONLINE" if online else "OFFLINE",
                      "syncing automatically" if online else "working locally",
                      "green" if online else "amber", icon="🌐",
                      badge=("LIVE", "g") if online else ("LOCAL", "a")),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("Pending", q["pending"], "saved on disk",
                      "amber" if q["pending"] else "", icon="📥"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("In flight", q["in_flight"], "claimed, awaiting ack",
                      icon="⏳"), unsafe_allow_html=True)
    c[3].markdown(kpi("Delivered", q["done"], "processed once (idempotent)",
                      "green", icon="✅"), unsafe_allow_html=True)
    c[4].markdown(kpi("Failed", q["failed"], "kept for manual requeue",
                      "red" if q["failed"] else "", icon="⚠️"),
                  unsafe_allow_html=True)


# ------------------------------------------------------------- live power
def live_panel():
    r = d.live_power()
    hist = st.session_state.setdefault("live_hist", [])
    hist.append({"time": dt.datetime.now(), "CPU %": r["cpu"],
                 "RAM %": r["ram"], "Battery W": r["watts"]})
    del hist[:-60]
    c = st.columns(4)
    if r["watts"] is not None:
        c[0].markdown(kpi("Battery power now", f"{r['watts']:.1f} W",
                          "measured (discharge-rate counter, lagged)", "blue",
                          icon="🔌", badge=("MEASURED", "g")),
                      unsafe_allow_html=True)
    else:
        c[0].markdown(kpi("Battery power now", "n/a",
                          "plugged in or no battery: not measurable, "
                          "nothing is estimated here", "amber", icon="🔌"),
                      unsafe_allow_html=True)
    c[1].markdown(kpi("CPU", f"{r['cpu']:.0f} %", "psutil, measured",
                      icon="🧠", badge=("LIVE", "b")), unsafe_allow_html=True)
    c[2].markdown(kpi("RAM", f"{r['ram']:.0f} %", "psutil, measured",
                      icon="💾", badge=("LIVE", "b")), unsafe_allow_html=True)
    bp = r["battery_pct"]
    c[3].markdown(kpi("Battery", "n/a" if bp is None else f"{bp:.0f} %",
                      "plugged in" if r["plugged"] else "on battery",
                      "green" if r["plugged"] else "amber", icon="🔋",
                      badge=("PLUGGED IN", "g") if r["plugged"]
                      else ("ON BATTERY", "a")),
                  unsafe_allow_html=True)
    if len(hist) > 2:
        df = pd.DataFrame(hist)
        long = df.melt("time", value_vars=["CPU %", "RAM %"],
                       var_name="metric", value_name="percent")
        enc = dict(
            x=alt.X("time:T", title=None),
            y=alt.Y("percent:Q", title="% utilisation",
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("metric:N", title=None,
                            scale=alt.Scale(domain=["CPU %", "RAM %"],
                                            range=[BLUE, GREEN])))
        line = alt.Chart(long).mark_line(interpolate="monotone",
                                         strokeWidth=2.5).encode(**enc)
        dots = alt.Chart(long.groupby("metric").tail(1)).mark_point(
            filled=True, size=90).encode(**enc)
        st.altair_chart((line + dots).properties(height=200),
                        use_container_width=True)
        st.caption(f"Last sample {df['time'].iloc[-1]:%H:%M:%S}. "
                   "CPU and RAM are measured with psutil.")
        if df["Battery W"].notna().any():
            st.markdown("**Battery power (W), measured**")
            st.line_chart(df.set_index("time")[["Battery W"]], height=140)
        else:
            st.info("Battery power is not available (plugged in or no battery), "
                    "so it is not plotted. Nothing is estimated here.")


if hasattr(st, "fragment"):                      # auto-refresh
    status_strip = st.fragment(run_every=3)(status_strip)
    live_panel = st.fragment(run_every=5)(live_panel)
status_strip()
st.write("")

tab_live, tab_res, tab_ai, tab_inf, tab_q, tab_m = st.tabs(
    ["⚡ Live power", "📊 Experiments", "🤖 AI insights",
     "🔋 Net savings", "🔄 Sync queue", "📖 Methodology"])

# -------------------------------------------------------------- live power
with tab_live:
    st.write("Live readings from this laptop. Battery watts are shown only "
             "when unplugged; the discharge-rate counter updates slowly, so "
             "totals in the experiments use the capacity drop instead.")
    live_panel()

# ---------------------------------------------------------------- results
with tab_res:
    facts = build()
    if not facts["scenarios"]:
        st.info("No experiment runs yet. Run: python harness.py run background")
    else:
        best = max(facts["scenarios"], key=lambda s: s["saving_percent"])
        kind = ("SYNTHETIC workload" if best["scenario"] == "background"
                else "real workload")
        st.markdown(
            '<div class="action"><div class="h">Recommended next action</div>'
            f'Apply the <b>{best["scenario"]}</b> optimization: measured energy '
            f'saving {best["saving_percent"]}% ({best["saving_wh"]} Wh per '
            f'{best["window_seconds"]:.0f} s window, {kind}). '
            'Written by code from measured data.</div>', unsafe_allow_html=True)
        st.markdown("#### Savings with 95% confidence interval")
        st.altair_chart(ci_chart(facts["scenarios"]), use_container_width=True)
        st.caption("A bar whose interval does not touch zero is distinguishable "
                   "from noise at 95%.")

    for s in facts["scenarios"]:
        synth = s["scenario"] == "background"
        st.markdown(f"### {s['scenario'].capitalize()}  "
                    + (pill("SYNTHETIC", "a") if synth else pill("REAL WORKLOAD", "g"))
                    + pill("ENERGY MEASURED", "b"), unsafe_allow_html=True)
        st.caption(d.SCENARIO_LABEL.get(s["scenario"], s["description"]))
        m = st.columns(4)
        m[0].metric("Baseline", f"{s['baseline_wh']} Wh")
        m[1].metric("Optimized", f"{s['optimized_wh']} Wh",
                    delta=f"-{s['saving_wh']} Wh ({s['saving_percent']}%)",
                    delta_color="inverse")
        m[2].metric("Cost saving", f"{s['cost_saving']}",
                    help="currency units per window = saving_Wh / 1000 x tariff")
        m[3].metric("CO2e saving", f"{s['co2e_saving_g']} g",
                    help="saving_Wh / 1000 x grid factor (g/kWh)")
        lo, hi = s["ci95_wh"]
        verdict = ("statistically distinguishable from noise (95%)"
                   if s["statistically_significant"]
                   else "NOT significant: indicative only")
        st.write(f"Per {s['window_seconds']:.0f} s window, {s['paired_repeats']} "
                 f"paired repeats, {s['pairs_favouring_optimized']} of "
                 f"{s['paired_repeats']} pairs favour optimized. "
                 f"95% CI of saving: {lo} to {hi} Wh. **{verdict}.**")
        st.write("")

    runs = d.run_energies()
    if runs:
        st.markdown("#### Every measured run (Wh per window)")
        df = pd.DataFrame(runs)
        cols = st.columns(2)
        for col, sc in zip(cols, ("background", "batching")):
            sub = df[df.scenario == sc]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="seed", columns="variant",
                                  values="energy_wh", aggfunc="mean")
            piv.index = piv.index.astype(str)
            with col:
                st.caption(sc.capitalize())
                st.altair_chart(run_chart(piv), use_container_width=True)

# ----------------------------------------------------------- AI explanations
with tab_ai:
    st.write("The model only phrases numbers computed in code. Every number in "
             "its reply is checked against the data; if it fails, the code "
             "writes a safe fallback text.")
    force = st.checkbox("Demo: queue as a NEW item even if the data is unchanged")
    if st.button("Generate explanations with the local SLM (about 20 s)",
                 type="primary"):
        with st.spinner("Local model is writing..."):
            f = build()
            rows, queued = save(f, explain_all(f), force_new=force)
        st.success(f"Saved {rows} new insight(s), queued {queued} for sync. "
                   "(0 means the same data was already explained.)")
    items = d.insights()
    if not items:
        st.info("No insights saved yet. Click the button above.")
    for it in items:
        badge = {"slm": pill("SLM: numbers verified", "g"),
                 "prediction": pill("prediction", "b")}.get(
                     it["source"], pill("template fallback", "a"))
        qcls = {"done": "g", "pending": "a"}.get(it["queue"], "x")
        with st.container(border=True):
            st.markdown(f"**{it['scenario'].capitalize()}**  {badge}"
                        + pill("LOCAL AI: offline", "b")
                        + pill(f"queue: {it['queue']}", qcls),
                        unsafe_allow_html=True)
            st.write(f"**Summary:** {it.get('summary')}")
            st.write(f"**Recommendation:** {it.get('recommendation')}")
            with st.expander("Evidence and method"):
                st.write(f"**Reason:** {it.get('reason')}")
                st.caption(f"Evidence (written by code, not the model): "
                           f"{it.get('evidence')}")
                st.caption(f"Saved {it['created_at']} | id {it['rec_id']}")
                st.code(f"{it.get('summary')} {it.get('recommendation')}",
                        language=None)      # the code box has a copy button

# ------------------------------------------------ inference & net savings
with tab_inf:
    inf = d.inference_summary()
    if inf is None:
        st.info("Run: python benchmark_slm_energy.py (unplugged)")
    else:
        st.write("Same prompts for every variant of qwen2.5:1.5b. Energy is "
                 "whole-system battery drop, minus idle power. Measured.")
        st.dataframe(pd.DataFrame(inf["table"]), use_container_width=True)
        iw = inf["idle_w"]
        idle_note = (f"idle power varied {min(iw):.1f} to {max(iw):.1f} W "
                     "between repeats. " if iw else "")
        st.caption(
            f"Source: {inf['file']}, {inf['reps']} repeats. Energy differences "
            f"between models are NOT statistically established; {idle_note}"
            "Q2_K was fast but could not follow the output schema, so its low "
            "energy buys nothing. Q4_K_M is the model used in this app.")
        m, sd = inf["copilot_wh"], inf["copilot_sd"]
        if m:
            st.markdown("#### Net savings: what the copilot costs vs what it saves")
            cols = st.columns(2)
            for col, s in zip(cols, build()["scenarios"]):
                sv, win = s["saving_wh"], s["window_seconds"]
                with col:
                    st.markdown(kpi(
                        f"{s['scenario'].capitalize()}: net saving",
                        f"{sv - m:.3f} Wh",
                        f"gross {sv} Wh, copilot {m:.4f} Wh "
                        f"({100 * m / sv:.1f}%). Break-even "
                        f"{m / sv * win:.0f} s (worst case "
                        f"{(m + sd) / sv * win:.0f} s)", "green", icon="🔋"),
                        unsafe_allow_html=True)
            st.caption("Indicative only: 3 repeats; the background scenario "
                       "is synthetic.")

# -------------------------------------------------------------- sync queue
with tab_q:
    st.write("Delivery is at-least-once; the receiver dedupes by idempotency key, "
             "so the effect is processed once. Items survive restarts and crashes.")
    st.dataframe(pd.DataFrame(d.recent_outbox()), use_container_width=True)

# ----------------------------------------------------------- method & limits
with tab_m:
    f = build()
    a = f["assumptions"]
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown(f"""
**Energy measurement:** battery capacity drop (Windows WMI) over each 300 s run,
sampled every 1 s. The discharge-rate integral is a cross-check only.
Measured, not modelled.

**Formulas:** `Wh = capacity drop` | `cost = kWh x tariff` | `CO2e = kWh x grid factor`

**Assumptions:** tariff = {a['tariff_per_kwh']} currency units/kWh,
grid factor = {a['grid_g_per_kwh']} g CO2e/kWh.

**Statistics:** paired t-test; baseline and optimized share a seed (identical
work); ABBA ordering cancels slow drift.
""")
    with c2:
        with st.container(border=True):
            st.markdown("""
**Limits:**
- One Windows laptop. Results are not general.
- The background app is a synthetic stand-in, not a real application.
- Battery capacity has coarse resolution; the rate counter is lagged and can
  differ from the capacity drop by tens of percent in a single run.
- The number verifier checks that numbers exist in the data, not that wording
  is sensible.
- Inference energy is whole-system, net of idle, from 3 repeats; model-to-model
  energy differences are not statistically established.
- The J/token figures are not comparable with published per-token numbers.
""")