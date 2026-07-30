"""
Tests for the real-data ingestion pipeline (connectors + ingest).

NOTE: the GraphQL response below is a SYNTHETIC FIXTURE used only to validate the
PARSING/ASSEMBLY CODE offline. The numbers are arbitrary and are NOT chemistry
data -- they exist solely to exercise the logic without hitting the live API.
The live fetch itself runs on an open network (see examples/fetch_real_data.py).
"""
import os, tempfile
from co2dash.connectors import (_parse_reactions_response, _node_to_record,
                                _decode_species, save_dataset, load_dataset)
from co2dash.ingest import (build_descriptor_table, descriptor_coverage,
                            assemble_candidates, _normalize_species)

# --- synthetic GraphQL response (shape only; values arbitrary) ----------------
FIXTURE = {
    "reactions": {
        "edges": [
            {"node": {"id": "1", "Equation": "CO2 + * -> COOH*",
                      "reactants": '{"CO2gas": 1, "star": 1}',
                      "products": '{"COOHstar": 1}', "reactionEnergy": 0.42,
                      "surfaceComposition": "Cu", "facet": "111",
                      "dftFunctional": "BEEF-vdW", "pubId": "pubA"}},
            {"node": {"id": "2", "Equation": "CO2 + * -> CO*",
                      "reactants": '{"CO2gas": 1}', "products": '{"COstar": 1}',
                      "reactionEnergy": -0.10, "surfaceComposition": "Cu",
                      "facet": "111", "dftFunctional": "BEEF-vdW", "pubId": "pubA"}},
            {"node": {"id": "3", "products": '{"OCHOstar": 1}',
                      "reactionEnergy": 0.05, "surfaceComposition": "Ag",
                      "facet": "100", "dftFunctional": "RPBE", "pubId": "pubB"}},
        ],
        "pageInfo": {"hasNextPage": False, "endCursor": "xyz"},
    }
}


def test_parse_reactions_response():
    nodes, has_next, cursor = _parse_reactions_response(FIXTURE)
    assert len(nodes) == 3
    assert has_next is False
    assert cursor == "xyz"


def test_decode_species_handles_json_strings():
    d = _decode_species('{"COstar": 1, "star": 1}')
    assert d["COstar"] == 1
    assert _decode_species("") == {}
    assert _decode_species({"a": 1}) == {"a": 1}


def test_normalize_species():
    assert _normalize_species("COstar") == "co"
    assert _normalize_species("CO*") == "co"
    assert _normalize_species("CO2gas") == "co2"
    assert _normalize_species("OCHOstar") == "ocho"


def test_node_to_record_is_tier_tagged():
    nodes, _, _ = _parse_reactions_response(FIXTURE)
    rec = _node_to_record(nodes[0])
    assert rec["tier"] == "COMPUTED"
    assert rec["surface"] == "Cu" and rec["facet"] == "111"
    assert "COOHstar" in rec["products"]
    assert "catalysis-hub:BEEF-vdW" in rec["source"]


def test_build_descriptor_table_groups_by_surface():
    nodes, _, _ = _parse_reactions_response(FIXTURE)
    records = [_node_to_record(n) for n in nodes]
    table = build_descriptor_table(records)
    # Cu(111) has both CO and COOH -> 2 intermediates; Ag(100) has OCHO -> 1
    cu = next(r for r in table if r["surface"] == "Cu")
    assert cu["n_intermediates"] == 2
    assert abs(cu["dE_CO"] - (-0.10)) < 1e-9
    assert abs(cu["dE_COOH"] - 0.42) < 1e-9
    assert cu["dE_OCHO"] is None
    # table is sorted most-complete first
    assert table[0]["n_intermediates"] >= table[-1]["n_intermediates"]


def test_assemble_candidates_skips_incomplete():
    nodes, _, _ = _parse_reactions_response(FIXTURE)
    records = [_node_to_record(n) for n in nodes]
    table = build_descriptor_table(records)
    # require both CO and COOH -> only Cu(111) qualifies, Ag(100) is skipped
    cands = assemble_candidates(table, ["dE_CO", "dE_COOH"])
    assert len(cands) == 1
    assert "Cu" in cands[0].material_id


def test_dataset_roundtrip():
    nodes, _, _ = _parse_reactions_response(FIXTURE)
    records = [_node_to_record(n) for n in nodes]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "chub.json")
        save_dataset(records, p, meta={"products": "COstar"})
        blob = load_dataset(p)
        assert blob["meta"]["source"] == "catalysis-hub"
        assert blob["meta"]["products"] == "COstar"
        assert len(blob["records"]) == 3
