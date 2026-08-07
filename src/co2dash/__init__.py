"""co2dash: zero-new-DFT techno-economic-environmental platform for CO2 utilisation.

Architecture (data flows top to bottom, decisions flow back up):
    connectors  -> public DFT/energy/grid data, tier-tagged
    config      -> YAML loader + ProvenanceRegistry (piece 1)
    surrogate   -> descriptors -> KPI (mean, std)   [swap-in slot for KAN/BNN]
    calibration -> calibrate/conformalise the surrogate's uncertainty (piece 3)
    techno_economic -> KPI -> LCOP / net-abatement / MAC  (scalar + vectorised, piece 2)
    uncertainty -> vectorised MC propagation + Sobol global sensitivity (piece 2)
    active_learning -> EVOI: which candidate to compute next
"""
from .schema import (Quantity, DataTier, Reaction, Candidate,
                     REACTIONS, RXN_METHANOL, RXN_FORMATE, RXN_CO)
from .techno_economic import (Scenario, evaluate_array,
                              marginal_abatement_cost_array)
from .surrogate import BayesianLinearSurrogate, cv_noise_precision
from .uncertainty import propagate_mc, sobol_indices
from .active_learning import rank_candidates
from .config import load_scenario, ProvenanceRegistry
from .calibration import (coverage_report, miscalibration_area,
                          TemperatureScaler, SplitConformal, CalibratedSurrogate)

__all__ = ["Quantity", "DataTier", "Reaction", "Candidate", "REACTIONS",
           "RXN_METHANOL", "RXN_FORMATE", "RXN_CO", "Scenario",
           "evaluate_array", "marginal_abatement_cost_array",
           "BayesianLinearSurrogate", "cv_noise_precision",
           "propagate_mc", "sobol_indices",
           "rank_candidates", "load_scenario", "ProvenanceRegistry",
           "coverage_report", "miscalibration_area", "TemperatureScaler",
           "SplitConformal", "CalibratedSurrogate"]

# feasibility envelope (the keystone 2-lever viability map)
from .uncertainty import feasibility_envelope          # noqa: E402
__all__.append("feasibility_envelope")

# ingestion of real public DFT data (runs on open network)
from .connectors import (fetch_catalysis_hub_reactions, probe_schema,        # noqa: E402
                         raw_graphql, save_dataset, load_dataset)
from .ingest import (build_descriptor_table, descriptor_coverage,            # noqa: E402
                     assemble_candidates, ingest_co2rr, DEFAULT_INTERMEDIATES)
__all__ += ["fetch_catalysis_hub_reactions", "probe_schema", "raw_graphql", "save_dataset",
            "load_dataset", "build_descriptor_table", "descriptor_coverage",
            "assemble_candidates", "ingest_co2rr", "DEFAULT_INTERMEDIATES"]

# physics activity proxy (CHE limiting potential -> cell voltage target)
from .proxy import (limiting_potential, build_activity_targets, proxy_cell_voltage,  # noqa: E402
                    PATHWAYS, EQUILIBRIUM_POTENTIALS, TYPICAL_ZPE_TS_CORRECTIONS)
__all__ += ["limiting_potential", "build_activity_targets", "proxy_cell_voltage",
            "PATHWAYS", "EQUILIBRIUM_POTENTIALS", "TYPICAL_ZPE_TS_CORRECTIONS"]

# energy/grid: choose a country/region (sourced static profiles + live connectors)
from .energy import (CountryProfile, COUNTRY_PROFILES, list_regions,         # noqa: E402
                     get_energy, apply_to_scenario,
                     fetch_opennem_au, fetch_electricitymaps)
__all__ += ["CountryProfile", "COUNTRY_PROFILES", "list_regions", "get_energy",
            "apply_to_scenario", "fetch_opennem_au", "fetch_electricitymaps"]

# user-facing tool layer: data intake (CSV -> validated Scenario) + recommendations
from .defaults import defaults_for, GENERIC_DEFAULTS, CONVENTIONAL_PRICE   # noqa: E402
from .intake import (map_columns, row_to_scenario, read_csv, ingest_table,   # noqa: E402
                     IntakeResult, COLUMN_ALIASES)
from .recommend import recommend, Recommendation
from .validation import (validate_energy, PRODUCTS as JOUNY_PRODUCTS, JOUNY_BASE, JOUNY_OPT)                          # noqa: E402
__all__ += ["defaults_for", "GENERIC_DEFAULTS", "CONVENTIONAL_PRICE",
            "map_columns", "row_to_scenario", "read_csv", "ingest_table",
            "IntakeResult", "COLUMN_ALIASES", "recommend", "Recommendation",
            "validate_energy", "JOUNY_PRODUCTS", "JOUNY_BASE", "JOUNY_OPT"]

# calibration gate: rigorous train/cal/test harness + procedure validation
from .calibration_harness import (calibrate_and_evaluate, CalibrationReport,
                                  split_indices, join_labeled,
                                  make_linear_synthetic, ConstStdSurrogate)
__all__ += ["calibrate_and_evaluate", "CalibrationReport", "split_indices",
            "join_labeled", "make_linear_synthetic", "ConstStdSurrogate"]

# public CO2RR experimental corpus featurizer (real-data calibration substitute)
from .corpus import featurize_co2rr, map_corpus_columns  # noqa: E402
__all__ += ["featurize_co2rr", "map_corpus_columns"]

# discovery<->decision join: link experimental FE to DFT descriptors
from .link import (canonical_material, availability_report,
                   descriptor_request_list, link_fe_to_descriptors,
                   descriptors_to_canonical)  # noqa: E402
__all__ += ["canonical_material","availability_report",
            "descriptor_request_list","link_fe_to_descriptors","descriptors_to_canonical"]

# adaptive loader for public CO2RR DFT datasets (Figshare etc.)
from .loaders import (load_table, resolve_columns, species_of,
                      to_descriptor_activity)  # noqa: E402
__all__ += ["load_table","resolve_columns","species_of","to_descriptor_activity"]

# data-quality guards (honest behaviour on low-quality inputs)
from .quality import data_quality_report, QualityReport  # noqa: E402
__all__ += ["data_quality_report","QualityReport"]

# HEA multi-sheet DFT workbook: per-configuration intermediate join (CO/CHO/COOH)
from .hea import (load_workbook, load_sheet, join_intermediates,          # noqa: E402
                  pathway_coverage, to_activity_table, decode_site,
                  assert_no_leakage, check_energy_reference,
                  che_reference_shift, to_che_formation_energies,
                  convert_rows_to_che, SPECIES_COMPOSITION,
                  SheetData, JoinReport, ELEMENT_BY_DESCRIPTOR)
__all__ += ["load_workbook", "load_sheet", "join_intermediates",
            "pathway_coverage", "to_activity_table", "decode_site",
            "assert_no_leakage", "check_energy_reference",
            "che_reference_shift", "to_che_formation_energies",
            "convert_rows_to_che", "SPECIES_COMPOSITION", "SheetData",
            "JoinReport", "ELEMENT_BY_DESCRIPTOR"]

# composition -> descriptors (the user-facing material interface)
from .composition import (Composition, DESCRIPTOR_BY_ELEMENT, ELEMENTS,   # noqa: E402
                          feature_names, sample_configurations,
                          configurations_to_descriptors,
                          descriptors_for_composition,
                          align_to_training_columns, sro_note)
__all__ += ["Composition", "DESCRIPTOR_BY_ELEMENT", "ELEMENTS",
            "feature_names", "sample_configurations",
            "configurations_to_descriptors", "descriptors_for_composition",
            "align_to_training_columns", "sro_note"]

# the discovery->decision chain: composition -> E_ads -> V_cell -> Scenario
from .chain import (ChainProvenance, ChainResult, EnsemblePrediction,     # noqa: E402
                    IntermediateModel, ReferenceFrame, REFERENCE_MODES,
                    train_intermediate_models, predict_composition,
                    apply_reference, run_chain, rank_compositions,
                    pds_uniform, applicability_report, SPREAD_ALARM_RATIO,
                    DFT, ASSUMED)
__all__ += ["ChainProvenance", "ChainResult", "EnsemblePrediction",
            "IntermediateModel", "ReferenceFrame", "REFERENCE_MODES",
            "train_intermediate_models", "predict_composition",
            "apply_reference", "run_chain", "rank_compositions",
            "pds_uniform", "applicability_report", "SPREAD_ALARM_RATIO",
            "DFT", "ASSUMED"]
