"""
Connectors to PUBLIC data sources. Federation, not scraping.

The Catalysis-Hub connector runs on an open network. Two robustness features
learned from the live API:
  * arguments that are null are OMITTED from the query (the filter resolver
    crashes with a 500 if passed null reactants/products/after);
  * requested fields are intersected with the live schema (self-healing against
    field-name drift), using GraphQL introspection.
It NEVER fabricates data: a failed call raises.
"""
from __future__ import annotations
import json, os, time
from typing import List, Dict, Optional
from .schema import Quantity, DataTier

try:
    import requests
except Exception:
    requests = None

CATALYSIS_HUB_GRAPHQL = "https://api.catalysis-hub.org/graphql"

# Preferred fields per reaction node. Intersected with the live schema at runtime,
# so requesting a non-existent field is harmless (it is dropped).
REACTION_FIELDS = ("id", "Equation", "reactants", "products", "reactionEnergy",
                   "activationEnergy", "surfaceComposition", "facet",
                   "dftCode", "dftFunctional", "chemicalComposition", "pubId")

_SCHEMA_CACHE: dict = {}


def _require_requests():
    if requests is None:
        raise RuntimeError("`requests` is required: pip install requests")


def _post_graphql(query: str, variables: dict | None = None, timeout: int = 60) -> dict:
    _require_requests()
    r = requests.post(CATALYSIS_HUB_GRAPHQL,
                      json={"query": query, "variables": variables or {}},
                      timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "errors" in payload and payload["errors"]:
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")
    return payload["data"]


def raw_graphql(query: str, variables: dict | None = None) -> dict:
    """Public escape hatch for experimenting with the API by hand."""
    return _post_graphql(query, variables)


def probe_schema(type_name: str = "Reaction") -> List[str]:
    """Introspect the live schema and return the field names of `type_name`."""
    q = "query($n:String!){ __type(name:$n){ fields{ name } } }"
    data = _post_graphql(q, {"n": type_name})
    t = data.get("__type") or {}
    return [f["name"] for f in (t.get("fields") or [])]


def _available_fields(type_name: str = "Reaction") -> Optional[set]:
    if type_name not in _SCHEMA_CACHE:
        try:
            _SCHEMA_CACHE[type_name] = set(probe_schema(type_name))
        except Exception:
            _SCHEMA_CACHE[type_name] = None   # introspection failed -> use as-is
    return _SCHEMA_CACHE[type_name]


def _valid_fields() -> tuple:
    avail = _available_fields("Reaction")
    if not avail:
        return REACTION_FIELDS
    fields = tuple(f for f in REACTION_FIELDS if f in avail)
    return fields or ("id", "reactionEnergy", "products", "surfaceComposition", "facet")


def _build_query(fields: tuple, use_after: bool, use_products: bool,
                 use_reactants: bool) -> str:
    """Build a query that DECLARES AND USES ONLY the arguments that have values.
    Null filter arguments crash the Catalysis-Hub resolver (500)."""
    decl = ["$first:Int!"]
    use = ["first:$first"]
    if use_after:     decl.append("$after:String");     use.append("after:$after")
    if use_products:  decl.append("$products:String");  use.append("products:$products")
    if use_reactants: decl.append("$reactants:String"); use.append("reactants:$reactants")
    body = "\n            ".join(fields)
    return ("query(%s){ reactions(%s){ edges { node {\n            %s\n        } } "
            "pageInfo { hasNextPage endCursor } } }"
            % (", ".join(decl), ", ".join(use), body))


def _parse_reactions_response(data: dict):
    """(nodes, has_next, end_cursor) -- pure, unit-testable without network."""
    reactions = (data or {}).get("reactions") or {}
    edges = reactions.get("edges") or []
    nodes = [e.get("node", {}) for e in edges if e.get("node")]
    page = reactions.get("pageInfo") or {}
    return nodes, bool(page.get("hasNextPage")), page.get("endCursor")


def _decode_species(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {"_raw": raw}
    return {}


def _node_to_record(nd: dict) -> dict:
    func = nd.get("dftFunctional") or nd.get("dftCode") or ""
    return {
        "reaction_id": nd.get("id") or nd.get("pubId") or "",
        "equation": nd.get("Equation") or nd.get("equation") or "",
        "reactants": _decode_species(nd.get("reactants")),
        "products": _decode_species(nd.get("products")),
        "reaction_energy": nd.get("reactionEnergy"),
        "activation_energy": nd.get("activationEnergy"),
        "surface": nd.get("surfaceComposition") or nd.get("chemicalComposition") or "",
        "facet": nd.get("facet") or "",
        "tier": DataTier.COMPUTED.name,
        "source": f"catalysis-hub:{func}:{nd.get('pubId','')}",
    }


def fetch_catalysis_hub_reactions(products: str = "",
                                  reactants: str = "",
                                  max_records: int = 500,
                                  page_size: int = 50,
                                  cache_path: Optional[str] = None,
                                  force: bool = False,
                                  validate_fields: bool = True) -> List[Dict]:
    """
    Fetch reaction/adsorption-energy records from Catalysis-Hub with pagination
    and optional on-disk caching. Filter by species name (e.g. products="COstar").
    Runs on an open network; raises on failure (never fabricates).
    """
    if cache_path and os.path.exists(cache_path) and not force:
        return load_dataset(cache_path)["records"]

    fields = _valid_fields() if validate_fields else REACTION_FIELDS
    out: list[dict] = []
    cursor = None
    while len(out) < max_records:
        first = min(page_size, max_records - len(out))
        query = _build_query(fields, use_after=cursor is not None,
                             use_products=bool(products), use_reactants=bool(reactants))
        variables = {"first": first}
        if cursor is not None: variables["after"] = cursor
        if products:           variables["products"] = products
        if reactants:          variables["reactants"] = reactants

        data = _post_graphql(query, variables)
        nodes, has_next, cursor = _parse_reactions_response(data)
        out.extend(_node_to_record(n) for n in nodes)
        if not has_next or not nodes:
            break
        time.sleep(0.2)   # be polite

    if cache_path:
        save_dataset(out, cache_path,
                     meta={"products": products, "reactants": reactants,
                           "fields": list(fields), "fetched_n": len(out)})
    return out


# --------------------------------------------------------------------- caching
def save_dataset(records: List[Dict], path: str, meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    blob = {"meta": {"source": "catalysis-hub",
                     "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})},
            "records": records}
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)


def load_dataset(path: str) -> Dict:
    with open(path, "r") as fh:
        return json.load(fh)


# --------------------------------------------------------------- other sources
def fetch_materials_project(formula: str, api_key: Optional[str] = None) -> List[Dict]:
    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise RuntimeError("Install mp-api: pip install mp-api") from exc
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "energy_above_hull", "band_gap"])
    return [{"material_id": str(d.material_id),
             "descriptors": {"band_gap": d.band_gap, "e_above_hull": d.energy_above_hull},
             "source_db": "materials-project", "tier": DataTier.COMPUTED.name,
             "source": f"mp:{d.material_id}"} for d in docs]


def grid_intensity_quantity(value_kg_per_kwh: float, source: str = "user-provided") -> Quantity:
    return Quantity(value=value_kg_per_kwh, tier=DataTier.LIT_EXTRACTED,
                    unit="kgCO2/kWh", source=source)
