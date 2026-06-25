"""
extractors/rule_graph_builder.py
--------------------------------
Builds the Knowledge Graph DETERMINISTICALLY from the extracted rulebook Rule
nodes — no LLM. The KG is a clean visual representation of the rules:

    (rulebook doc) -[:contains parameter]-> (parameter) -[<constraint>]-> (value)

e.g.  (RITES DPR Appraisal Tool) -[contains parameter]-> (Design Axle Load)
                                  -[must be at least]->   (25 tons)

The graph reuses the existing Entity/TRIPLE schema (Entity nodes + TRIPLE
relationships with a `relation` property) so the Neo4j Browser query
`MATCH (h:Entity)-[r:TRIPLE]->(t:Entity) RETURN h,r,t`, the FAISS index builder,
and get_kg_stats all work unchanged.

Source of truth: the structured Rule nodes created by run_push.py (which come
from rule_extractor.py). This is built from the RULES DOCUMENT ONLY.
"""

import hashlib

from loguru import logger

from config.settings import NodeLabel, RelType
from utils.neo4j_client import run_read, run_write


def _eid(*parts) -> str:
    """Stable entity id from its text parts (so repeated params/values dedup)."""
    return hashlib.md5("::".join(str(p) for p in parts).lower().encode()).hexdigest()[:16]


def _constraint_phrase(operator: str) -> str:
    """Human-readable TRIPLE relation for the parameter→value edge."""
    return {
        ">=": "must be at least",
        ">":  "must exceed",
        "<=": "must not exceed",
        "<":  "must be under",
        "==": "must equal",
        "in_range": "must be within",
        "must_be": "must be",
        "requires": "requires",
    }.get(str(operator), "has value")


# ─── Semantic layer: group parameters into engineering-domain categories ───────
# Hybrid by design: keyword rules give a precise (syntactic) category where we're
# confident; anything left over is sent to the LLM for concept induction (semantic).
_CATEGORY_KEYWORDS: dict[str, tuple] = {
    "Track Geometry":          ("gradient", "curve", "radius", "gauge", "alignment",
                                 "formation", "width", "track centre", "track center", "spacing", "cant"),
    "Permanent Way":           ("rail", "sleeper", "ballast", "uts", "weight", "track structure"),
    "Operations":              ("speed", "headway", "capacity", "ridership", "traffic", "tkm", "rkm", "km"),
    "Electrification & Traction": ("traction", "scada", "ohe", "substation", "sectioning",
                                   "paralleling", "tower wagon", "sp", "ssp", "tss"),
    "Signalling":              ("signal", "interlocking", "block", "ei "),
    "Structures & Loading":    ("axle", "loading", "load", "bridge", "viaduct", "span",
                                "foundation", "pier", "deck", "seismic"),
    "Cost & Schedule":         ("cost", "crore", "lakh", "schedule", "completion", "period"),
}


def _keyword_category(attr: str) -> "str | None":
    a = str(attr).lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(k in a for k in kws):
            return cat
    return None


def _llm_categorize(attrs: list[str]) -> dict:
    """One batched LLM call to assign a short engineering-domain category to each
    parameter the keyword rules couldn't place. Returns {attr: category}."""
    if not attrs:
        return {}
    try:
        from utils.llm_router import generate_json
        listing = "\n".join(f"- {a}" for a in attrs)
        prompt = (
            "Assign each railway DPR parameter below to ONE short engineering-domain "
            "category (2-3 words, e.g. 'Track Geometry', 'Electrification', 'Structures', "
            "'Operations', 'Signalling', 'Permanent Way', 'Cost & Schedule').\n\n"
            f"Parameters:\n{listing}\n\n"
            'Return ONLY a JSON object: {"parameter": "Category", ...}'
        )
        res = generate_json(prompt, temperature=0.0)
        return {k: v for k, v in res.items()} if isinstance(res, dict) else {}
    except Exception as e:
        logger.debug(f"LLM categorisation unavailable: {e}")
        return {}


def build_rule_graph(rulebook_doc_id: str, sector: str = None, clear: bool = True,
                     add_categories: bool = True) -> dict:
    """
    Build the rule KG for a rulebook from its Rule nodes.

    Args:
        rulebook_doc_id: the rulebook Document doc_id (e.g. "41ba21bf_rules").
        sector:          restrict to one sector's rules (None = all rules).
        clear:           wipe any prior Entity/TRIPLE for this doc first.

    Returns: {"entities", "triples", "parameters", "values"} counts.
    """
    if clear:
        run_write(f"MATCH (e:{NodeLabel.ENTITY} {{doc_id: $id}}) DETACH DELETE e",
                  {"id": rulebook_doc_id})

    where = "WHERE rs.name = $sector" if sector else ""
    rules = run_read(
        f"""
        MATCH (r:{NodeLabel.RULE})-[:{RelType.BELONGS_TO}]->(rs:{NodeLabel.SECTOR})
        {where}
        RETURN r.attribute AS attr, r.threshold AS thr, r.unit AS unit,
               r.operator AS op, r.severity AS sev, r.clause AS clause,
               r.condition AS cond, r.standard_name AS std
        """,
        {"sector": sector} if sector else {},
    )
    if not rules:
        logger.warning(f"No rules found for rulebook {rulebook_doc_id} (sector={sector}).")
        return {"entities": 0, "triples": 0, "parameters": 0, "values": 0}

    doc_label = next((r["std"] for r in rules if r.get("std")), "Rulebook")
    doc_eid   = _eid("doc", doc_label)

    # ── Semantic layer: categorise each parameter (keyword-first, LLM for the rest)
    cat_map: dict = {}
    if add_categories:
        unique_attrs = sorted({(r.get("attr") or "").strip() for r in rules if (r.get("attr") or "").strip()})
        uncategorized = []
        for a in unique_attrs:
            c = _keyword_category(a)
            if c:
                cat_map[a] = c
            else:
                uncategorized.append(a)
        cat_map.update(_llm_categorize(uncategorized))   # semantic fallback
        logger.info(f"Categorised {len(cat_map)} of {len(unique_attrs)} parameters "
                    f"({len(unique_attrs) - len(uncategorized)} by keyword, "
                    f"{len(uncategorized)} via LLM)")

    params, values, categories, triples = set(), set(), set(), 0

    for r in rules:
        attr = (r.get("attr") or "").strip()
        if not attr:
            continue
        thr  = (str(r.get("thr") or "")).strip()
        unit = (r.get("unit") or "").strip()
        value_label = f"{thr} {unit}".strip() or "(specified)"
        rel = _constraint_phrase(r.get("op"))

        param_eid = _eid("param", sector or "", attr)
        value_eid = _eid("value", attr, value_label)   # value scoped to its parameter
        params.add(param_eid)
        values.add(value_eid)

        run_write(
            f"""
            MERGE (d:{NodeLabel.DOCUMENT} {{doc_id: $rb}})
            MERGE (doc_e:{NodeLabel.ENTITY} {{entity_id: $doc_eid}})
              ON CREATE SET doc_e.label = $doc_label, doc_e.doc_id = $rb,
                            doc_e.sector = $sector, doc_e.kind = 'document'
            MERGE (d)-[:{RelType.HAS_ENTITY}]->(doc_e)

            MERGE (p:{NodeLabel.ENTITY} {{entity_id: $param_eid}})
              ON CREATE SET p.label = $attr, p.doc_id = $rb,
                            p.sector = $sector, p.kind = 'parameter'
            MERGE (d)-[:{RelType.HAS_ENTITY}]->(p)

            MERGE (v:{NodeLabel.ENTITY} {{entity_id: $value_eid}})
              ON CREATE SET v.label = $value_label, v.doc_id = $rb,
                            v.sector = $sector, v.kind = 'value'
            MERGE (d)-[:{RelType.HAS_ENTITY}]->(v)

            MERGE (doc_e)-[:{RelType.TRIPLE} {{relation: 'contains parameter'}}]->(p)
            MERGE (p)-[hv:{RelType.TRIPLE} {{relation: $rel}}]->(v)
              SET hv.operator = $op, hv.severity = $sev,
                  hv.clause = $clause, hv.condition = $cond
            """,
            {
                "rb": rulebook_doc_id, "doc_eid": doc_eid, "doc_label": doc_label,
                "param_eid": param_eid, "attr": attr,
                "value_eid": value_eid, "value_label": value_label,
                "rel": rel, "op": r.get("op", ""), "sev": r.get("sev", ""),
                "clause": r.get("clause", ""), "cond": r.get("cond", ""),
                "sector": sector or "",
            },
        )
        triples += 2  # contains-parameter + parameter-value

        # Semantic edge: parameter -[in category]-> domain concept
        category = cat_map.get(attr)
        if category:
            cat_eid = _eid("cat", category)
            categories.add(cat_eid)
            run_write(
                f"""
                MERGE (d:{NodeLabel.DOCUMENT} {{doc_id: $rb}})
                MERGE (c:{NodeLabel.ENTITY} {{entity_id: $cat_eid}})
                  ON CREATE SET c.label = $category, c.doc_id = $rb,
                                c.sector = $sector, c.kind = 'category'
                MERGE (d)-[:{RelType.HAS_ENTITY}]->(c)
                WITH c
                MATCH (p:{NodeLabel.ENTITY} {{entity_id: $param_eid}})
                MERGE (p)-[:{RelType.TRIPLE} {{relation: 'in category'}}]->(c)
                """,
                {"rb": rulebook_doc_id, "cat_eid": cat_eid, "category": category,
                 "param_eid": param_eid, "sector": sector or ""},
            )
            triples += 1

    n_entities = 1 + len(params) + len(values) + len(categories)
    logger.success(
        f"Rule graph built for {rulebook_doc_id}: {len(params)} parameters, "
        f"{len(values)} values, {len(categories)} categories, {triples} edges"
    )
    return {"entities": n_entities, "triples": triples, "parameters": len(params),
            "values": len(values), "categories": len(categories)}
