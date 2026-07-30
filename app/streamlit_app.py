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
                     miscalibration_area, TemperatureScaler)
from co2dash.techno_economic import (capital_recovery_factor,
                                     specific_electricity_kwh_per_kg)
from co2dash.energy import list_regions, get_energy, apply_to_scenario
from co2dash.intake import map_columns, ingest_table
from co2dash.recommend import recommend
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
RX = {"Methanol (6e⁻)": RXN_METHANOL, "Formate (2e⁻)": RXN_FORMATE, "CO (2e⁻)": RXN_CO}

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Scenario levers")
    rxn = RX[st.selectbox("Product", list(RX))]
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
    st.divider()
    uploaded = st.file_uploader("Load tier-tagged scenario (YAML)", type=["yaml", "yml"])

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
    st.sidebar.success("Loaded from YAML (overrides sliders).")
else:
    base = slider_scenario()

# region override (applies to both slider- and YAML-built scenarios)
if region_code is not None:
    _e = get_energy(region_code)
    base = apply_to_scenario(base, region_code)
    _q = _e["grid_intensity"]
    st.sidebar.caption(f"Grid: **{_e['name']}** — {_q.value:.3f} kgCO₂/kWh  \nSource: {_q.source}")

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

# --------------------------------------------------------------------- header + verdict
st.markdown('<div class="hero-eyebrow">CO₂ utilisation · techno-economic & environmental</div>'
            '<div class="hero-title">Feasibility readout</div>'
            '<div class="hero-sub">Zero new DFT · public data + uncertainty-aware translation layer · '
            'all defaults are illustrative placeholders.</div>', unsafe_allow_html=True)

p_feas, p_net = mc["p_mac_below_carbon_price"], mc["p_net_positive"]
if p_net < 0.5:
    status, vc = "Not climate-positive", RED
elif p_feas >= 0.5:
    status, vc = "Feasible", GREEN
elif p_feas >= 0.15:
    status, vc = "Marginal", AMBER
else:
    status, vc = "Not feasible at this price", RED
mac_med_t = "∞" if not np.isfinite(mc["mac_median"]) else f"{mc['mac_median']*1000:,.0f}"
st.markdown(f"""
<div class="verdict" style="--vc:{vc}">
  <div class="verdict-status">{status}</div>
  <div class="verdict-stats">
    <div><div class="vstat-label">P(MAC &lt; price)</div><div class="vstat-value">{p_feas:.0%}</div></div>
    <div><div class="vstat-label">P(net &gt; 0)</div><div class="vstat-value">{p_net:.0%}</div></div>
    <div><div class="vstat-label">MAC median</div><div class="vstat-value">{mac_med_t} $/t</div></div>
  </div>
</div>""", unsafe_allow_html=True)

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
with st.expander("🧭 Recommended next steps (plain-language synthesis)", expanded=False):
    st.caption("Runs the MC verdict, Sobol sensitivity, breakeven and target search, "
               "then tells you what to improve, by how much, and what to compute next.")
    if st.button("Generate recommendation", key="rec_btn"):
        with st.spinner("Analysing…"):
            rec = recommend(base, carbon_price, registry=registry, n_mc=20_000)
        for s in rec.steps:
            st.markdown(f"- {s}")

# --------------------------------------------------------------------- tabs
t1, t2, t3, t4, t5, t6 = st.tabs(["Economics & climate", "Feasibility envelope",
                                  "Sensitivity", "Active learning", "Calibration",
                                  "Your data"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cost build-up (LCOP)**")
        st.plotly_chart(ui.cost_waterfall(comp_fixed, comp_co2, comp_elec, comp_h2),
                        use_container_width=True)
        st.markdown('<span class="cap">Where each dollar of product cost comes from. '
                    'Electricity usually dominates energy-intensive routes.</span>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown("**MAC distribution (Monte-Carlo)**")
        st.plotly_chart(ui.mac_distribution(mc["mac"], carbon_price, mc["mac_p05"],
                                            mc["mac_median"], mc["mac_p95"]),
                        use_container_width=True)
        st.markdown(f'<span class="cap">Shaded = feasible draws (MAC &lt; carbon price). '
                    f'Median {mc["mac_median"]:.2f}, P05 {mc["mac_p05"]:.2f}, '
                    f'P95 {mc["mac_p95"]:.2f} $/kg.</span>', unsafe_allow_html=True)
    if registry is not None:
        st.markdown("**Provenance registry** — every number, its uncertainty, and its source")
        st.dataframe(registry.table(), use_container_width=True, hide_index=True)
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
                        use_container_width=True)
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
                    use_container_width=True)
    top = names[0]
    st.markdown(f'<span class="cap">Highest total-order term: <b>{top}</b> — pin this down '
                f'first. Low S1 with high ST means it acts through interactions.</span>',
                unsafe_allow_html=True)

with t4:
    st.markdown("**Which catalyst to compute next?** Ranked by expected value of "
                "information toward feasibility.")
    st.markdown('<span class="cap">Demo uses synthetic descriptors; replace with real '
                'Catalysis-Hub / OC20 data and your calibrated surrogate.</span>',
                unsafe_allow_html=True)
    rng = np.random.default_rng(0)
    keys = ["dE_CO", "dE_COOH"]
    Xtr = rng.uniform([-1.5, -1.0], [0.5, 1.5], size=(50, 2))
    ytr = 0.9 * np.exp(-((Xtr[:, 0] + 0.6) ** 2 + (Xtr[:, 1] - 0.2) ** 2)) + rng.normal(0, 0.05, 50)
    surr = BayesianLinearSurrogate(degree=2).fit(Xtr, ytr)
    n_cand = st.slider("Number of candidate materials", 5, 30, 12)
    cands = [Candidate(material_id=f"cand_{i}",
                       descriptors={"dE_CO": float(x[0]), "dE_COOH": float(x[1])})
             for i, x in enumerate(rng.uniform([-1.5, -1.0], [0.5, 1.5], size=(n_cand, 2)))]
    ranked = rank_candidates(cands, surr, keys, base, carbon_price, seed=2)
    st.dataframe(ranked, use_container_width=True, hide_index=True)

with t5:
    st.markdown("**Is the surrogate's uncertainty trustworthy?** Reliability of a demo model.")
    rng = np.random.default_rng(1)
    mean = rng.normal(0, 2, 4000); y = mean + rng.normal(0, 1.0, 4000)
    over = st.slider("Reported std (true spread = 1.0)", 0.2, 2.0, 0.4, 0.1)
    std = np.full_like(mean, over)
    levels = (0.5, 0.8, 0.9, 0.95)
    before = coverage_report(mean, std, y, levels)
    ts = TemperatureScaler().fit(mean, std, y); m2, s2 = ts.transform(mean, std)
    after = coverage_report(m2, s2, y, levels)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(ui.reliability_diagram(list(levels),
                        [before[l] for l in levels], [after[l] for l in levels]),
                        use_container_width=True)
    with c2:
        st.metric("Temperature scale s", f"{ts.s:.2f}",
                  help=">1 inflates an over-confident std; <1 shrinks under-confident")
        st.metric("Miscalibration", f"{miscalibration_area(mean, std, y):.3f}",
                  delta=f"{miscalibration_area(m2, s2, y) - miscalibration_area(mean, std, y):+.3f}",
                  delta_color="inverse")
        st.markdown('<span class="cap">Points on the dotted line = perfectly calibrated. '
                    'Below it = over-confident.</span>', unsafe_allow_html=True)

# --------------------------------------------------------------------- Your data
with t6:
    st.markdown("**Check your own experiments / calculations**")
    st.caption("Upload a CSV of your measurements. Recognised columns (any of): "
               "material, product, FE (or %), cell voltage (V or mV), current density, "
               "electricity price, grid intensity. Unmeasured inputs are filled from "
               "sourced defaults and flagged. One row = one scenario.")
    default_rxn = st.selectbox("Default product (used when a row has no 'product' column)",
                               ["co", "methanol", "formate"], key="ud_rxn")
    up = st.file_uploader("Measurements CSV", type=["csv"], key="ud_csv")

    with st.expander("No file? Try a small example"):
        st.code("material,product,FE (%),cell voltage,current density\n"
                "Ag-foam,CO,92,3.2,250\nAg-NP,CO,88,3.1,180\nCu-oxide,CO,74,3.6,120",
                language="text")

    if up is not None:
        text = up.getvalue().decode("utf-8")
        import csv as _csv, io as _io
        headers = next(_csv.reader(_io.StringIO(text)))
        mapping, unknown = map_columns(headers)
        st.markdown(f"**Recognised columns:** {', '.join(f'`{k}`→{v}' for k, v in mapping.items()) or '—'}")
        if unknown:
            st.caption(f"Ignored (not used): {', '.join(unknown)}")

        results = ingest_table(text, default_rxn)
        table = []
        for i, res in enumerate(results):
            ev = res.scenario.evaluate()
            mac = ev["mac_usd_per_tonne_co2"]
            table.append({
                "row": i,
                "FE": round(res.scenario.faradaic_efficiency, 3),
                "V_cell": round(res.scenario.cell_voltage, 2),
                "net kg/kg": round(ev["net_abatement_kg_per_kg"], 3),
                "MAC $/t": "∞" if not np.isfinite(mac) else f"{mac:,.0f}",
                "status": "ok" if res.ok else "errors",
                "flags": len(res.warnings) + len(res.errors),
            })
        st.dataframe(table, use_container_width=True, hide_index=True)

        idx = st.number_input("Inspect row", 0, max(0, len(results) - 1), 0, key="ud_row")
        res = results[idx]
        if res.errors:
            for e in res.errors:
                st.error(e)
        for w in res.warnings:
            st.warning(w)
        with st.expander("Provenance (user vs sourced default)"):
            for k, v in res.provenance.items():
                st.markdown(f"- `{k}`: {v}")
        if st.button("Recommended next steps for this row", key="ud_rec"):
            with st.spinner("Analysing…"):
                rec = recommend(res.scenario, carbon_price, n_mc=20_000)
            for s in rec.steps:
                st.markdown(f"- {s}")
