# co2dash — example files (what to load where)

| File | Load in… | What it is | Provenance |
|---|---|---|---|
| `scenario_co_real.yaml` | Sidebar → "Load tier-tagged scenario (YAML)" | A full, sourced CO2→CO scenario (performance + economics + environment) | Literature / public databases; every field carries a tier + source |
| `your_data_example.csv` | Tab "Your data" | Six real Ag→CO experiments (measured FE, voltage, current density) | Osorio-Tejada et al. 2024, Table 2 (public) |
| ACS Catalysis HEA `.xlsx` (`cs2c03675_si_002.xlsx`) | Sidebar → "Real DFT descriptors" → tabs "Calibration" / "Active learning" | DFT adsorption energies (*CO/*CHO/*COOH) on FeCoNiCuMo HEA | ACS Catalysis 2023 SI — download from Figshare (NOT bundled: it is third-party data) |

Notes
- The YAML and CSV here are public/synthetic and safe to ship as demo assets.
- The HEA .xlsx is third-party data: users download it themselves from Figshare
  (ACS Catalysis, article cs2c03675). Do not commit it to the repo.
- "Your data" fills any economics you didn't provide with sourced literature
  defaults, marked "default: <source>" vs "user".
