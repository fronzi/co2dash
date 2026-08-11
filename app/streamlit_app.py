"""
co2dash interactive GUI (Streamlit) — instrument-readout edition.

Visual identity: a laboratory-instrument panel. Electrochemistry teal, a
carbon/energy amber, monospaced data readouts, and a feasibility verdict as the
page thesis. A thin wrapper: every number comes from co2dash.* pure functions.

Run:  streamlit run app/streamlit_app.py     (needs: pip install -e ".[ui]")
All defaults are illustrative placeholders — load a tier-tagged YAML for real data.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st

from co2dash import (RXN_METHANOL, RXN_FORMATE, RXN_CO, Scenario, propagate_mc,
                     sobol_indices, feasibility_envelope, BayesianLinearSurrogate,
                     rank_candidates, Candidate, load_scenario, coverage_report,
                     miscalibration_area, TemperatureScaler, calibrate_and_evaluate)
from co2dash.techno_economic import (capital_recovery_factor,
                                     specific_electricity_kwh_per_kg)
from co2dash.energy import list_regions, get_energy, apply_to_scenario
from co2dash.intake import map_columns, ingest_table
from co2dash.recommend import recommend
from co2dash.composition import Composition, ELEMENTS, sro_note
from co2dash.chain import (ReferenceFrame, train_intermediate_models,
                           run_chain, rank_compositions, applicability_report,
                           DFT)
from co2dash.hea import load_workbook
import ui_charts as ui

st.set_page_config(page_title="co2dash · CO₂ utilisation platform",
                   layout="wide", page_icon="⬡")

# --------------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing:-0.01em; color:#14242E; }
.block-container { padding-top: 2.2rem; max-width: 1300px; }
.hero-eyebrow { font:700 13px 'Space Grotesk'; letter-spacing:.14em; text-transform:uppercase; color:#0E7C86; padding-top:1rem; margin-bottom:.15rem; }
.hero-title { font:700 34px 'Space Grotesk'; color:#14242E; letter-spacing:-.01em; margin:0 0 .35rem; line-height:1.15; }
.hero-sub { font:400 14px Inter, sans-serif; color:#6B7A82; margin-bottom:.6rem; line-height:1.45; max-width:70ch; }
/* verdict strip */
.verdict { display:flex; align-items:center; gap:20px; border:1px solid #E3E8EC;
  border-left:6px solid var(--vc); border-radius:12px; padding:14px 20px; background:#fff;
  box-shadow:0 1px 2px rgba(20,36,46,.04); margin:.4rem 0 1.1rem; }
.verdict-status { font:700 20px 'Space Grotesk'; color:var(--vc); }
.verdict-stats { display:flex; gap:28px; margin-left:auto; }
.vstat-label { font:600 10px 'Space Grotesk'; letter-spacing:.12em; text-transform:uppercase; color:#5C6B73; }
.vstat-value { font:600 18px 'JetBrains Mono'; color:#14242E; }
/* kpi cards */
.kpi-row { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:.6rem; }
.kpi { position:relative; background:#fff; border:1px solid #E3E8EC; border-radius:12px;
  padding:14px 16px 18px; overflow:hidden; box-shadow:0 1px 2px rgba(20,36,46,.04); }
.kpi-label { font:600 10.5px 'Space Grotesk'; letter-spacing:.13em; text-transform:uppercase; color:#5C6B73; }
.kpi-value { font:600 26px 'JetBrains Mono'; line-height:1.25; margin-top:4px; }
.kpi-unit { font:500 12px 'JetBrains Mono'; color:#5C6B73; margin-left:5px; }
.kpi-sub { font-size:11.5px; color:#5C6B73; margin-top:2px; min-height:14px; }
.kpi-bar { position:absolute; left:0; bottom:0; height:3px; width:100%; opacity:.85; }
.stTabs [data-baseweb="tab-list"] { gap:4px; }
.stTabs [data-baseweb="tab"] { font:600 13px 'Space Grotesk'; }
.cap { color:#5C6B73; font-size:12.5px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

GREEN, AMBER, RED, TEAL, INK = "#1E8E5A", "#C77D17", "#C0392B", "#0E7C86", "#14242E"
# CO first, deliberately: it is the default the app opens on. Methanol needs 6
# electrons and ~26 kWh/kg, so at the default sliders its net abatement is
# NEGATIVE and the app used to open on "Not climate-positive" — which reads as a
# broken dashboard rather than as a true statement about a hard route.
RX = {"CO (2e⁻)": RXN_CO, "Formate (2e⁻)": RXN_FORMATE, "Methanol (6e⁻)": RXN_METHANOL}

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Scenario levers")
    st.caption("Your hypotheses. A loaded YAML replaces all of these.")
    # NOTE: this offers "CO (2e-)" while the DFT section below offers "*CO".
    # They are different objects -- a product MOLECULE leaving the cell versus a
    # species ADSORBED on the surface -- so both carry the qualifier that makes
    # that unmistakable rather than a bare "CO".
    rxn = RX[st.selectbox("Product molecule (what the cell makes)", list(RX),
                          help="Sets the stoichiometry: electrons transferred, "
                               "molar mass, kg CO₂ consumed per kg of product.")]
    fe = st.slider("Faradaic efficiency", 0.05, 1.0, 0.60, 0.01,
                   help="Fraction of electrons going to the target product. Enters energy as 1/FE.")
    vcell = st.slider("Cell voltage (V)", 1.5, 5.0, 3.0, 0.1)
    c_elec = st.slider("Electricity price ($/kWh)", 0.01, 0.20, 0.06, 0.005)
    _regions = list_regions()
    _region_opts = ["Custom (use slider below)"] + list(_regions.values())
    _region_pick = st.selectbox("Grid region", _region_opts, index=0,
                                help="Pick a country/region to set grid intensity from sourced 2024 data. Overrides the slider.")
    region_code = None
    if _region_pick != _region_opts[0]:
        region_code = next(c for c, n in _regions.items() if n == _region_pick)
    grid = st.slider("Grid intensity (kgCO₂/kWh)", 0.0, 0.8, 0.05, 0.005,
                     disabled=region_code is not None,
                     help="Carbon intensity of the electricity. Drives the climate result.")
    phi = st.slider("End-of-life release φ", 0.0, 1.0, 0.10, 0.05,
                    help="0 = durable storage, 1 = fuel re-released. Lowers the storage credit.")
    capex = st.slider("CAPEX total ($M)", 10.0, 200.0, 50.0, 5.0) * 1e6
    carbon_price = st.slider("Carbon price ($/kg CO₂)", 0.0, 3.0, 0.30, 0.05,
                             help="Feasibility threshold the MAC is compared against.")
    # ------------------------------------------------------------ data inputs
    # All three uploads live here, together, each saying what it is FOR and what
    # it must CONTAIN. They were previously scattered — two in the sidebar, one
    # buried inside a tab — which made it hard to see that the app has three
    # independent data sources at all.
    st.divider()
    st.markdown("### Data inputs")
    st.caption("Three independent sources. Each drives different tabs; none is "
               "required to start.")

    st.markdown("**1 · Scenario** — assumptions")
    uploaded = st.file_uploader("Tier-tagged scenario (.yaml)",
                                type=["yaml", "yml"], key="yaml_up")
    st.caption("Plant, prices and grid, each with `value / std / tier / source`. "
               "Overrides the sliders above. Drives: Cost breakdown, Viability "
               "map, What matters most.")

    st.markdown("**2 · Your measurements** — experimental")
    ud_csv = st.file_uploader("Measurements (.csv)", type=["csv"], key="ud_csv")
    st.caption("One row per catalyst you measured: FE, cell voltage, current "
               "density. No model in between. Drives: Your measurements.")

    st.markdown("**3 · DFT descriptors** — computed")
    _dft_up = st.file_uploader("HEA workbook (.xlsx)", type=["xlsx"], key="dft")
    st.caption("One sheet per adsorbed intermediate (*CO / *CHO / *COOH), each "
               "row a site configuration plus its adsorption energy. Drives: "
               "Predict from composition, Next DFT to run, Model reliability.")
    _dft_int = st.selectbox(
        "Adsorbed intermediate — for 'Next DFT to run' & 'Model reliability' only",
        ["CO", "CHO", "COOH"], key="dft_int", format_func=lambda s: f"*{s}",
        help="A species bound to the surface — not the product molecule chosen "
             "above. *CO is CO adsorbed on the catalyst; CO the product is what "
             "leaves the cell. Those two tabs study one intermediate at a time. "
             "'Predict from composition' ignores this: the limiting potential "
             "needs *CO and *COOH together, so it loads every sheet.")

def slider_scenario():
    return Scenario(
        n_electrons=rxn.n_electrons, molar_mass_prod=rxn.molar_mass_prod,
        m_co2=rxn.kg_co2_per_kg_prod, m_h2=0.0,
        faradaic_efficiency=fe, cell_voltage=vcell,
        capex_total=capex, annual_production_kg=2.0e7, opex_fix_per_yr=3.0e6,
        disc_rate=0.08, lifetime_yr=20, c_co2=0.05, c_elec=c_elec, c_h2=0.0,
        lcop_conventional=0.40, grid_intensity=grid, e_capture=0.1, e_process=0.05,
        release_fraction=phi)

registry = None
if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="wb") as fh:
        fh.write(uploaded.getvalue()); tmp = fh.name
    base, registry = load_scenario(tmp)
    st.sidebar.success("Scenario loaded from YAML — every slider above, "
                       "including the product, is now inactive.")
else:
    base = slider_scenario()

# region override (applies to both slider- and YAML-built scenarios)
if region_code is not None:
    _e = get_energy(region_code)
    base = apply_to_scenario(base, region_code)
    _q = _e["grid_intensity"]
    st.sidebar.caption(f"Grid: **{_e['name']}** — {_q.value:.3f} kgCO₂/kWh  \nSource: {_q.source}")

# --------------------------------------------------- DFT-driven cell voltage
# Set from the 'Predict from composition' tab (session state, so it survives the rerun). Every
# performance field records its origin; the verdict strip renders that origin so
# an assumed input can never be mistaken for a predicted one.
KPI_ORIGIN = {"faradaic_efficiency": "assumed", "cell_voltage": "assumed"}
_chain_v = st.session_state.get("chain_v_cell")
_chain_label = st.session_state.get("chain_label", "")
_chain_unsourced = bool(st.session_state.get("chain_unsourced_anchor"))
if _chain_v is not None:
    import dataclasses as _dc
    base = _dc.replace(base, cell_voltage=float(_chain_v))
    KPI_ORIGIN["cell_voltage"] = "DFT"

# --------------------------------------------------------------------- compute
r = base.evaluate()
e_elec = specific_electricity_kwh_per_kg(base.n_electrons, base.cell_voltage,
                                         base.molar_mass_prod, base.faradaic_efficiency,
                                         base.rectifier_eff)
crf = capital_recovery_factor(base.disc_rate, base.lifetime_yr)
comp_fixed = (base.capex_total * crf + base.opex_fix_per_yr) / base.annual_production_kg
comp_co2 = base.c_co2 * base.m_co2
comp_elec = base.c_elec * e_elec
comp_h2 = base.c_h2 * base.m_h2

if registry is not None:
    unc = registry.mc_distributions(
        ["faradaic_efficiency", "cell_voltage", "c_elec", "capex_total", "grid_intensity"])
else:
    unc = {"faradaic_efficiency": ("normal", base.faradaic_efficiency, 0.10),
           "cell_voltage": ("normal", base.cell_voltage, 0.3),
           "c_elec": ("lognormal", base.c_elec, 1.4),
           "capex_total": ("lognormal", base.capex_total, 1.5),
           "grid_intensity": ("uniform", max(0.0, base.grid_intensity - 0.03),
                              base.grid_intensity + 0.03)}
mc = propagate_mc(base, unc, carbon_price, n=40_000, seed=0)

# --------------------------------------------------------------------- build stamp
# "Is the deployed app actually running the code I just pushed?" is otherwise
# unanswerable from the browser: Streamlit Community Cloud shows no commit, and a
# stale environment looks identical to a fresh one. Reading the checked-out SHA
# at run time makes it observable.
@st.cache_data(show_spinner=False)
def _build_stamp():
    import subprocess, datetime
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        when = subprocess.run(["git", "log", "-1", "--format=%cs"], cwd=root,
                              capture_output=True, text=True, timeout=5).stdout.strip()
        subj = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=root,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        sha = when = subj = ""
    if not sha:
        return "build: unknown (no git metadata in this deployment)"
    return f"build {sha} · {when} · {subj[:70]}"


# --------------------------------------------------------------------- header + verdict
st.markdown('<div class="hero-eyebrow">CO₂ utilisation · techno-economic & environmental</div>'
            '<div class="hero-title">Feasibility readout</div>'
            '<div class="hero-sub">Zero new DFT · public data + uncertainty-aware translation layer · '
            'all defaults are illustrative placeholders.</div>', unsafe_allow_html=True)
st.markdown(f'<span class="cap">{_build_stamp()}</span>', unsafe_allow_html=True)

p_feas, p_net = mc["p_mac_below_carbon_price"], mc["p_net_positive"]
_why = ""
if p_net < 0.5:
    status, vc = "Not climate-positive", RED
    # A bare red label reads as a broken dashboard. Name the binding constraint:
    # the electricity carbon almost always is it, and the breakeven says by how
    # much. Without this the only recourse is to move sliders at random.
    _bg = r.get("breakeven_grid_intensity", float("nan"))
    _why = (f"The plant emits more CO₂ than it stores: net abatement is "
            f"{r['net_abatement_kg_per_kg']:+.2f} kg per kg of product, because "
            f"it needs {e_elec:.1f} kWh/kg. ")
    if np.isfinite(_bg):
        _why += (f"Your grid is {base.grid_intensity:.3f} kgCO₂/kWh; it must be "
                 f"below **{_bg:.3f}** for this route to remove CO₂ at all. ")
    _why += ("Higher-electron products (methanol, 6e⁻) are the hardest — try "
             "CO (2e⁻), a cleaner grid, a lower cell voltage or a higher FE.")
elif p_feas >= 0.5:
    status, vc = "Feasible", GREEN
elif p_feas >= 0.15:
    status, vc = "Marginal", AMBER
else:
    status, vc = "Not feasible at this price", RED
mac_med_t = "∞" if not np.isfinite(mc["mac_median"]) else f"{mc['mac_median']*1000:,.0f}"
_hdr_subject = ("loaded YAML scenario" if registry is not None else "sidebar sliders")
st.markdown(f"""
<div class="verdict" style="--vc:{vc}">
  <div>
    <div class="vstat-label">Verdict for the {_hdr_subject}</div>
    <div class="verdict-status">{status}</div>
  </div>
  <div class="verdict-stats">
    <div><div class="vstat-label">P(MAC &lt; price)</div><div class="vstat-value">{p_feas:.0%}</div></div>
    <div><div class="vstat-label">P(net &gt; 0)</div><div class="vstat-value">{p_net:.0%}</div></div>
    <div><div class="vstat-label">MAC median</div><div class="vstat-value">{mac_med_t} $/t</div></div>
  </div>
</div>""", unsafe_allow_html=True)

if _why:
    st.warning(_why)

# provenance strip: which KPIs are model-driven and which are assumed
_dft_kpis = sorted(k for k, v in KPI_ORIGIN.items() if v == "DFT")
_assumed = sorted(k for k, v in KPI_ORIGIN.items() if v != "DFT")
if _chain_unsourced:
    st.error(
        "**These numbers are not quotable.** The cell voltage driving this "
        "verdict comes from a limiting potential fixed by an **unsourced anchor**. "
        "The anchor sets a constant added to every U_L, so the MAC shown here "
        "moves rigidly with a value nobody has cited. Rankings between "
        "compositions are unaffected; the absolute figures are not defensible "
        "until you supply a cited anchor U_L, or your own gas-phase reference "
        "energies in 'absolute' mode.")
if _dft_kpis:
    st.markdown(
        f'<span class="cap">Verdict provenance — <b>DFT-driven:</b> '
        f'{", ".join(_dft_kpis)} ({_chain_label}). <b>Assumed:</b> '
        f'{", ".join(_assumed)}. Faradaic efficiency is never predicted: no '
        f'descriptor→selectivity model exists.</span>', unsafe_allow_html=True)
else:
    st.markdown(
        '<span class="cap">Verdict provenance — <b>no KPI is DFT-driven.</b> '
        'This readout is a function of your slider/YAML inputs only. Use the '
        '<b>Composition</b> tab to drive cell voltage from the HEA descriptors.'
        '</span>', unsafe_allow_html=True)

# --------------------------------------------------------------------- KPI row
def kpi(label, value, unit, accent, sub=""):
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{accent}">{value}'
            f'<span class="kpi-unit">{unit}</span></div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'<div class="kpi-bar" style="background:{accent}"></div></div>')

net = r["net_abatement_kg_per_kg"]
mac_t = r["mac_usd_per_tonne_co2"]
bg = r.get("breakeven_grid_intensity", float("nan"))
grid_ok = np.isfinite(bg) and base.grid_intensity < bg
cards = [
    kpi("LCOP", f"{r['lcop_usd_per_kg']:.2f}", "$/kg", INK, "levelized product cost"),
    kpi("Energy intensity", f"{e_elec:.1f}", "kWh/kg", AMBER, "electricity per kg product"),
    kpi("Net abatement", f"{net:+.2f}", "kg/kg", GREEN if net > 0 else RED,
        "CO₂ removed per kg" if net > 0 else "emits more than it stores"),
    kpi("MAC", "∞" if not np.isfinite(mac_t) else f"{mac_t:,.0f}", "$/t CO₂",
        INK if np.isfinite(mac_t) else RED, "marginal abatement cost"),
    kpi("Breakeven grid", "—" if not np.isfinite(bg) else f"{bg:.3f}", "kg/kWh",
        GREEN if grid_ok else AMBER, "stay below to be climate-positive"),
]
st.markdown('<div class="kpi-row">' + "".join(cards) + '</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------- next steps
SCENARIO_LABEL = ("the loaded YAML scenario" if registry is not None
                  else "the sidebar slider settings")

with st.expander(f"🧭 Recommended next steps — for {SCENARIO_LABEL}", expanded=False):
    st.caption(f"Elaborates the headline verdict above, which describes "
               f"**{SCENARIO_LABEL}**. Adds the Sobol sensitivity, the breakeven "
               f"grid and the target value each lever must reach. The 'Your measurements' "
               f"tab has its own separate verdict, one per uploaded row.")
    if st.button("Generate recommendation", key="rec_btn"):
        with st.spinner("Analysing…"):
            rec = recommend(base, carbon_price, registry=registry, n_mc=20_000,
                            subject=SCENARIO_LABEL)
        for s in rec.steps:
            st.markdown(f"- {s}")

# --------------------------------------------------------------------- tabs
@st.cache_data(show_spinner=False)
def _load_hea(file_bytes, sheet):
    """Real DFT descriptor loader: HEA .xlsx (CO/CHO/COOH sheets) -> (X, y, labels).
    X = elemental site features, y = Eads (eV) of the chosen intermediate."""
    import pandas as pd, io
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    target = "Eads (eV)" if "Eads (eV)" in df.columns else df.columns[-1]
    feats = [c for c in df.columns if c not in ("Labels", target)]
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(float)
    # the 'Labels' column here duplicates Eads (not a material name), so identify
    # each alloy configuration by its dataset row index instead.
    labels = [f"config #{i}" for i in range(len(df))]
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[m], y[m], [l for l, keep in zip(labels, m) if keep]


_dft = None
if _dft_up is not None:
    try:
        _dft = _load_hea(_dft_up.getvalue(), _dft_int)
    except Exception as _e:
        st.sidebar.error(f"Could not read descriptors: {_e}")

# Tabs are grouped by WHICH INPUT DRIVES THEM, because that was the recurring
# confusion: three different data sources were interleaved with nothing saying so.
#   1-3  the scenario (YAML or sliders) — always available
#   4    your experimental CSV
#   5-7  the DFT workbook — inactive until you upload it
# The unpacking order below re-orders the display without moving any code: the
# `with tN:` blocks further down keep their original numbering.
t1, t2, t3, t6, t7, t4, t5 = st.tabs([
    "Cost breakdown",            # t1 — scenario
    "Viability map",             # t2 — scenario
    "What matters most",         # t3 — scenario
    "Your measurements",         # t6 — your CSV
    "Predict from composition",  # t7 — DFT workbook
    "Next DFT to run",           # t4 — DFT workbook
    "Model reliability",         # t5 — DFT workbook
])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cost build-up (LCOP)**")
        st.plotly_chart(ui.cost_waterfall(comp_fixed, comp_co2, comp_elec, comp_h2),
                        width='stretch')
        st.markdown('<span class="cap">Where each dollar of product cost comes from. '
                    'Electricity usually dominates energy-intensive routes.</span>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown("**MAC distribution (Monte-Carlo)**")
        st.plotly_chart(ui.mac_distribution(mc["mac"], carbon_price, mc["mac_p05"],
                                            mc["mac_median"], mc["mac_p95"]),
                        width='stretch')
        st.markdown(f'<span class="cap">Shaded = feasible draws (MAC &lt; carbon price). '
                    f'Median {mc["mac_median"]:.2f}, P05 {mc["mac_p05"]:.2f}, '
                    f'P95 {mc["mac_p95"]:.2f} $/kg.</span>', unsafe_allow_html=True)
    if registry is not None:
        st.markdown("**Provenance registry** — every number, its uncertainty, and its source")
        st.dataframe(registry.table(), width='stretch', hide_index=True)
    else:
        st.info("Load a YAML scenario to see the provenance table "
                "(value · uncertainty · tier · source) and tier-derived uncertainty.")

with t2:
    st.markdown("**Where does a viable industry exist?** Sweep two levers; "
                "the white line is the feasibility boundary (MAC = carbon price).")
    AX = {"Faradaic efficiency": ("faradaic_efficiency", 0.3, 0.95),
          "Cell voltage (V)": ("cell_voltage", 2.0, 4.5),
          "Electricity price ($/kWh)": ("c_elec", 0.02, 0.18),
          "Grid intensity (kgCO₂/kWh)": ("grid_intensity", 0.005, 0.2),
          "CAPEX total ($)": ("capex_total", 1e7, 1.5e8)}
    cc1, cc2 = st.columns(2)
    xlab = cc1.selectbox("X axis", list(AX), index=0)
    ylab = cc2.selectbox("Y axis", list(AX), index=3)
    if AX[xlab][0] == AX[ylab][0]:
        st.warning("Choose two different axes.")
    else:
        xf, x0, x1 = AX[xlab]; yf, y0, y1 = AX[ylab]
        env = feasibility_envelope(base, xf, np.linspace(x0, x1, 60),
                                   yf, np.linspace(y0, y1, 60), carbon_price)
        st.plotly_chart(ui.envelope_heatmap(env["X"], env["Y"], env["mac"],
                                            env["feasible"], xlab, ylab),
                        width='stretch')
        st.metric("Viable fraction of swept region", f"{env['feasible'].mean():.0%}")

with t3:
    st.markdown("**Which lever moves the answer?** Global (Sobol) sensitivity of MAC.")
    bounds = {"faradaic_efficiency": (0.30, 0.95), "cell_voltage": (2.2, 4.0),
              "c_elec": (0.02, 0.15), "capex_total": (2e7, 1.2e8),
              "grid_intensity": (0.005, 0.2)}
    with st.spinner("Running Sobol…"):
        sob = sobol_indices(base, bounds, n=512)
    names = sorted(sob, key=lambda k: sob[k]["ST"], reverse=True)
    st.plotly_chart(ui.sobol_tornado(names, [sob[k]["S1"] for k in names],
                                     [sob[k]["ST"] for k in names]),
                    width='stretch')
    top = names[0]
    st.markdown(f'<span class="cap">Highest total-order term: <b>{top}</b> — pin this down '
                f'first. Low S1 with high ST means it acts through interactions.</span>',
                unsafe_allow_html=True)

with t4:
    st.markdown("**Which alloy to compute next?** Ranked by how much a new DFT "
                "calculation would cut the surrogate's uncertainty about the "
                "activity landscape.")
    if _dft is None:
        st.info("Upload a real DFT descriptor file in the sidebar (HEA .xlsx with "
                "CO/CHO/COOH sheets) to rank real candidates. This tab stays inactive "
                "until real descriptors are provided — no synthetic candidates.")
    else:
        import pandas as pd
        X, yv, labels = _dft
        Xk = X[:, X.std(0) > 1e-9]
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(yv))
        ntr = max(20, int(0.6 * len(yv)))
        tr, pool = idx[:ntr], idx[ntr:]
        # fit_evidence, not fit: the same model 'Predict from composition' uses. With a
        # hardcoded beta the reported sigma is a constructor default rather than
        # a result, and the "uncertainty (eV)" column below would be inflated.
        surr = BayesianLinearSurrogate().fit_evidence(Xk[tr], yv[tr])
        mean, sd = surr.predict(Xk[pool])
        order = np.argsort(-sd)
        tbl = pd.DataFrame({
            "alloy": [labels[pool[i]] for i in order],
            f"pred ΔE ·*{_dft_int} (eV)": np.round(mean[order], 3),
            "uncertainty (eV)": np.round(sd[order], 3),
        }).head(15)
        st.dataframe(tbl, width='stretch', hide_index=True)
        st.markdown(f'<span class="cap">Real HEA DFT · target = *{_dft_int} adsorption '
                    f'energy · trained on {len(tr)} configurations · {len(pool)} ranked '
                    f'by predictive uncertainty (most informative next calculation on '
                    f'top). Note these candidates are rows held out of THIS file, so '
                    f'their DFT answer already exists — the ranking demonstrates the '
                    f'loop rather than recommending unknown materials.</span>',
                    unsafe_allow_html=True)

with t5:
    st.markdown("**Is the surrogate's uncertainty trustworthy?** Reliability on real DFT data.")
    if _dft is None:
        st.info("Upload a real DFT descriptor file in the sidebar to see the real "
                "reliability diagram (train/calibration/test split on the uploaded "
                "data). No synthetic demo is shown.")
    else:
        X, yv, labels = _dft
        Xk = X[:, X.std(0) > 1e-9]
        rep = calibrate_and_evaluate(
            Xk, yv,
            surrogate_factory=lambda Xt, yt: BayesianLinearSurrogate().fit_evidence(Xt, yt),
            alpha=0.1, seed=0)
        lv = list(rep.levels)
        before = [rep.coverage_before[l] for l in lv]
        after = [rep.coverage_after[l] for l in lv]
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(ui.reliability_diagram(lv, before, after),
                            width='stretch')
        with c2:
            st.metric("Temperature scale s", f"{rep.temperature_s:.2f}",
                      help=">1 inflates an over-confident std; <1 shrinks under-confident. "
                           "s near 1 means the model was already honest and the gate "
                           "found nothing to correct — a pass, not a failure.")
            if abs(rep.temperature_s - 1.0) < 0.10:
                st.success("s ≈ 1: already calibrated, no correction needed.")
            st.metric("Miscalibration", f"{rep.miscal_before:.3f}",
                      delta=f"{rep.miscal_after - rep.miscal_before:+.3f}",
                      delta_color="inverse")
            st.markdown(f'<span class="cap">Real HEA DFT · *{_dft_int} · '
                        f'n={rep.n_train + rep.n_cal + rep.n_test}. On the dotted line '
                        f'= calibrated; below = over-confident; above = error bars '
                        f'wider than needed. Assessed on the same surrogate the '
                        f'"Predict from composition" tab uses (hyperparameters '
                        f'fitted by evidence '
                        f'maximisation), so this measures the uncertainty you '
                        f'actually consume.</span>', unsafe_allow_html=True)

# ------------------------------------------------------- Your measurements (CSV)
with t6:
    st.markdown("**Your measured catalysts, evaluated economically**")
    st.caption("Recognised columns (any of): material, product, FE (or %), cell "
               "voltage (V or mV), current density, electricity price, grid "
               "intensity. Unmeasured inputs are filled from your loaded scenario "
               "or from sourced defaults, and flagged. One row = one catalyst.")
    if ud_csv is None:
        st.info("Upload a **measurements .csv** in the sidebar (input 2) to use "
                "this tab.")
        with st.expander("What the file should look like"):
            st.code("material,product,FE (%),cell voltage,current density\n"
                    "Ag-foam,CO,92,3.2,250\nAg-NP,CO,88,3.1,180\nCu-oxide,CO,74,3.6,120",
                    language="text")

    if ud_csv is not None:
        text = ud_csv.getvalue().decode("utf-8")
        import csv as _csv, io as _io
        headers = next(_csv.reader(_io.StringIO(text)))
        mapping, unknown = map_columns(headers)
        st.markdown(f"**Recognised columns:** {', '.join(f'`{k}`→{v}' for k, v in mapping.items()) or '—'}")
        if unknown:
            st.caption(f"Ignored (not used): {', '.join(unknown)}")

        # The product fixes the stoichiometry (n, molar mass, kg CO2 per kg), so
        # it changes everything downstream. Only ask when the file does not say:
        # shown unconditionally it was inert for most files and quietly decisive
        # for the rest.
        if "product" in mapping.values():
            default_rxn = "co"          # unused: every row carries its own
            st.caption("Product read from your `product` column — the "
                       "stoichiometry comes from the file, not from a default.")
        else:
            st.warning(
                "Your file has no `product` column, so the reaction must be "
                "assumed — and it sets n, molar mass and kg CO₂ per kg. On an "
                "otherwise identical row this choice moves MAC from +954 (CO) to "
                "−343 $/t (formate). Add a `product` column to remove the guess.")
            default_rxn = st.selectbox("Assume this product for every row",
                                       ["co", "methanol", "formate"], key="ud_rxn")

        # pass the loaded scenario as the fill source: the CSV supplies what it
        # measured, the YAML supplies the plant and the grid. Without this the
        # rows were evaluated against generic defaults — a different plant from
        # the one driving the verdict above, silently.
        results = ingest_table(text, default_rxn, base=base if registry is not None else None)
        if registry is not None:
            st.caption("Unmeasured fields are taken from your loaded YAML scenario, "
                       "not from generic defaults. See the provenance panel below.")
        else:
            st.caption("No YAML loaded: unmeasured fields come from generic sourced "
                       "defaults. Load a scenario to evaluate your rows against your "
                       "own plant and grid instead.")
        table = []
        for i, res in enumerate(results):
            ev = res.scenario.evaluate()
            mac = ev["mac_usd_per_tonne_co2"]
            table.append({
                "material": res.material_id or f"(row {i})",
                "FE": round(res.scenario.faradaic_efficiency, 3),
                "V_cell": round(res.scenario.cell_voltage, 2),
                "net kg/kg": round(ev["net_abatement_kg_per_kg"], 3),
                "MAC $/t": "∞" if not np.isfinite(mac) else f"{mac:,.0f}",
                "status": "ok" if res.ok else "errors",
                "flags": len(res.warnings) + len(res.errors),
            })
        st.dataframe(table, width='stretch', hide_index=True)

        _names = [f"{r.material_id or '(unnamed)'} — row {i}"
                  for i, r in enumerate(results)]
        idx = _names.index(st.selectbox("Inspect catalyst", _names, key="ud_row"))
        res = results[idx]
        if res.errors:
            for e in res.errors:
                st.error(e)
        for w in res.warnings:
            st.warning(w)
        with st.expander("Provenance (user vs sourced default)"):
            for k, v in res.provenance.items():
                st.markdown(f"- `{k}`: {v}")
        _who = res.material_id or f"row {idx}"
        if st.button(f"Recommended next steps for {_who}", key="ud_rec"):
            with st.spinner("Analysing…"):
                rec = recommend(res.scenario, carbon_price, n_mc=20_000,
                                subject=f"{_who} (your row {idx})")
            for s in rec.steps:
                st.markdown(f"- {s}")

# --------------------------------------------------------------------- Composition
with t7:
    st.markdown("**Enter an alloy composition, not 40 descriptor columns**")
    st.caption("Descriptors are element properties, so they are derived from the "
               "composition via a lookup table. A composition specifies a "
               "distribution over site occupations, so the prediction is an "
               "ensemble, not a point.")

    if _dft is None:
        st.info("Upload the HEA descriptor workbook in the sidebar first — the "
                "surrogate is trained on it. Nothing here is synthetic.")
    else:
        _wb = st.session_state.get("_hea_sheets")
        if _wb is None:
            try:
                _wb = load_workbook(_dft_up.getvalue())
                st.session_state["_hea_sheets"] = _wb
            except Exception as _e:
                st.error(f"Could not load the workbook sheets: {_e}")
                _wb = None

        if _wb:
            st.caption(f"Trained per intermediate on: "
                       + ", ".join(f"*{k} (n={len(v)})" for k, v in sorted(_wb.items())))
            cmodels = train_intermediate_models(_wb)

            _appl = applicability_report(cmodels)
            _gaps = {k: v["missing_site1"] for k, v in _appl.items() if v["missing_site1"]}
            with st.expander("Applicability domain — what this workbook supports",
                             expanded=bool(_gaps)):
                st.dataframe(
                    [{"intermediate": f"*{k}", "n": v["n_train"],
                      "site-1 elements covered": ", ".join(v["site1_support"]),
                      "NOT covered": ", ".join(v["missing_site1"]) or "—",
                      "train sd (eV)": round(v["train_sd_eV"], 3)}
                     for k, v in _appl.items()],
                    width='stretch', hide_index=True)
                if _gaps:
                    st.warning(
                        "The sheets do not cover the same adsorption-site elements. "
                        "Predictions for a composition containing an uncovered "
                        "element are extrapolation, and the model's own error bar "
                        "does not detect it: the element appears in the environment "
                        "columns, so its descriptor values look in-range and the "
                        "novelty is only in the joint position, which a linear model "
                        "cannot see. Affected: "
                        + "; ".join(f"*{k} missing {', '.join(v)}"
                                    for k, v in _gaps.items()))

            c1, c2 = st.columns([2, 1])
            with c1:
                comp_text = st.text_input(
                    "Composition", value="FeCoNiCuMo",
                    help="e.g. FeCoNiCuMo (equimolar) or Fe0.4Co0.2Ni0.2Cu0.1Mo0.1")
            with c2:
                site1 = st.selectbox("Adsorption site element",
                                     ["(sampled)"] + ELEMENTS, key="c_site1")
            n_cfg = st.slider("Configurations sampled", 100, 2000, 500, 100)

            st.markdown("**Reference frame** — how adsorption energies become "
                        "CHE formation energies")
            mode = st.radio(
                "mode", ["relative", "anchored", "absolute"], horizontal=True,
                label_visibility="collapsed",
                help="relative: no absolute U_L claimed, ranking only (needs "
                     "nothing). anchored: one known U_L fixes the constant. "
                     "absolute: your own gas-phase total energies.")
            frame = None
            try:
                if mode == "anchored":
                    a1, a2 = st.columns(2)
                    e_anchor = a1.number_input("Anchor E_ads(*COOH) [eV]", value=-0.90,
                                               step=0.01, format="%.3f")
                    u_anchor = a2.number_input("Anchor known U_L [V vs RHE]", value=-0.45,
                                               step=0.01, format="%.3f")
                    src = st.text_input("Anchor source (cite it)", value="")
                    if not src.strip():
                        st.warning(
                            "No source given for the anchor. The anchor fixes a "
                            "constant added to every U_L, so an uncited value makes "
                            "every absolute MAC unquotable — the ranking survives, "
                            "the numbers do not.")
                    frame = ReferenceFrame(mode="anchored",
                                           anchor_energies={"COOH": e_anchor},
                                           anchor_U_L=u_anchor, anchor_source=src)
                elif mode == "absolute":
                    g1, g2, g3 = st.columns(3)
                    gas = {"CO2": g1.number_input("E(CO₂) [eV]", value=0.0, format="%.4f"),
                           "H2": g2.number_input("E(H₂) [eV]", value=0.0, format="%.4f"),
                           "H2O": g3.number_input("E(H₂O) [eV]", value=0.0, format="%.4f")}
                    st.caption("These must come from YOUR calculations — same code, "
                               "functional, pseudopotentials and cutoff as the slabs. "
                               "Another group's totals shift every U_L by an unknown "
                               "constant.")
                    frame = ReferenceFrame(mode="absolute", gas_energies=gas)
            except ValueError as _e:
                st.error(str(_e))
                frame = None

            if st.button("Predict", key="c_run"):
                try:
                    comp = Composition.from_string(comp_text)
                except ValueError as _e:
                    st.error(str(_e))
                    comp = None
                if comp is not None:
                    res = run_chain(comp, cmodels, base, frame,
                                    n_samples=int(n_cfg),
                                    fixed_site1=None if site1 == "(sampled)" else site1)
                    cols = st.columns(len(res.predictions))
                    for col, (sp, p) in zip(cols, sorted(res.predictions.items())):
                        _ood = p.warning() is not None
                        col.metric(f"ΔE *{sp}" + ("  ⚠︎" if _ood else ""),
                                   f"{p.mean:.3f} eV",
                                   help="ensemble mean over sampled configurations")
                        col.caption(f"configurational ±{p.configurational_sd:.3f} · "
                                    f"model ±{p.model_sd:.3f} eV")
                        if _ood:
                            col.error(p.warning())
                            if p.in_domain_mean is not None:
                                col.caption(
                                    f"in-domain only ({1 - p.out_of_domain_fraction:.0%} "
                                    f"of configs): {p.in_domain_mean:.3f} "
                                    f"± {p.in_domain_sd:.3f} eV")

                    for w in res.warnings:
                        st.error(w)

                    if res.v_cell is not None:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("U_L", f"{res.U_L:.3f} V")
                        m2.metric("PDS", res.pds)
                        m3.metric("V_cell (DFT)", f"{res.v_cell:.2f} V",
                                  help=f"±{res.v_cell_sd:.3f} eV from the surrogate")
                        st.session_state["chain_v_cell"] = res.v_cell
                        st.session_state["chain_label"] = f"{comp.label()}, {mode}"
                        st.session_state["chain_unsourced_anchor"] = (
                            mode == "anchored" and not (frame.anchor_source or "").strip())
                        st.success("Cell voltage pushed to the headline verdict. "
                                   "Rerun happens on the next interaction.")
                    else:
                        st.warning("No absolute cell voltage in 'relative' mode — "
                                   "ranking only. The headline verdict keeps your "
                                   "assumed V_cell.")
                        if res.relative_score is not None:
                            st.metric("Relative activity score",
                                      f"{res.relative_score:+.3f}",
                                      help="higher = more active; reference-free")

                    for n in res.provenance.notes:
                        st.caption(f"· {n}")

            if st.session_state.get("chain_v_cell") is not None:
                if st.button("Clear DFT-driven voltage (back to slider)", key="c_clear"):
                    st.session_state.pop("chain_v_cell", None)
                    st.session_state.pop("chain_label", None)
                    st.session_state.pop("chain_unsourced_anchor", None)
