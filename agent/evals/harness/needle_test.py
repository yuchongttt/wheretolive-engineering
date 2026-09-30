#!/usr/bin/env python3
"""Needle-in-haystack recall test for the Arden chat agent.

Pick a real property, write a STRICT but REALISTIC natural-language query from
its true attributes, hand it to the live agent, and check whether the agent
surfaces that exact property.

Two design rules that make this a fair test of the system's strengths:
  - NO postcode in the query. A real buyer says "within X minutes of <hub>",
    not "postcode SW11". So the agent must DERIVE the area via screen_by_commute
    / get_commute before it can search — retrieval, not a free outcode filter.
  - Lean on IMAGE-DERIVED data. The discriminating attributes come from the
    floorplan VLM (garden, reception-room count, bathrooms, total area) and the
    zero-shot decor score (interior finish percentile) — the parts of the
    pipeline that are this product's edge, not generic portal filters.

Two independent pass signals per needle:
  1. SEARCH RECALL — replay every search_properties / search_floorplans call the
     agent made (its args) against the DB using the SAME filter semantics the
     tools use, and check whether the target id is in the result set. Answers
     "did the agent construct a query that *can* find it?".
  2. SURFACED — scan the final answer for the target's listing id (it rides in
     the listing/image URLs) or its street name. Answers "did it show it?".

Runs against the real production endpoint via agent_eval.run_case.

Public copy: each needle pinned a real listing (its id and street). Those are
redacted to placeholders below; fill them from your own listings table to run.
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_eval import REPO, run_case, _admin_key  # noqa: E402

DB = REPO / "data" / "evaluations.db"

HOUSE = ('terraced', 'semi-detached', 'detached', 'end of terrace', 'house',
         'town house', 'link detached house', 'mews', 'cottage', 'bungalow',
         'barn conversion', 'semi-detached villa')
FLAT = ('flat', 'apartment', 'maisonette', 'studio', 'ground flat', 'penthouse',
        'duplex', 'triplex', 'ground maisonette', 'block of apartments')

# Three needles, each blind (no id, no street, NO postcode), each leaning on
# image-derived attributes. Counts below are at the time of authoring.
NEEDLES = [
    {
        # floorplan VLM: garden + 3 reception rooms + 3 baths. 2 such 4-bed
        # detached houses London-wide; only the target is <=40 min to Stratford.
        "name": "stratford-garden-house",
        "target_id": "<listing-id-1>",
        "street": "<street>",
        "query": ("I'm after a 4-bedroom detached freehold house with a private "
                  "garden and at least 3 reception rooms, plus 3 bathrooms, "
                  "within about 40 minutes commute of Stratford. Budget around "
                  "1.25 million (1.1m to 1.4m). No specific area in mind."),
    },
    {
        # decor-centric: top-percentile interior finish + balcony. The agent has
        # to use the decor score (screen_decor_value / get_decor) to rank, not a
        # portal filter. ~3 candidates near Canary Wharf in budget.
        "name": "canarywharf-decor-flat",
        "target_id": "<listing-id-2>",
        "street": "<street>",
        "query": ("Find me a 2-bed 2-bath apartment with a genuinely top-tier "
                  "interior finish - one of the best-presented flats you can find "
                  "- ideally with a balcony, within 15 minutes of Canary Wharf. "
                  "Budget around 875k, up to 950k."),
    },
    {
        # floorplan VLM + decor combined: garden + 3 receptions + top finish. 8
        # such 5-bed houses London-wide; only the target is <=20 min to Clapham
        # Junction.
        "name": "claphamjct-garden-decor-house",
        "target_id": "<listing-id-3>",
        "street": "<street>",
        "query": ("A large 5-bedroom freehold house with a garden and at least 3 "
                  "reception rooms, beautifully finished inside (top-tier decor), "
                  "within 20 minutes of Clapham Junction. Budget around 2.4 "
                  "million (2.2m to 2.6m)."),
    },
    {
        # floorplan VLM feature: garage (has_garage, rare in London). Unique
        # after commute — only the target is a 3-bed freehold house with a
        # garage <=30 min of Clapham Junction under 550k.
        "name": "garage-house-claphamjct",
        "target_id": "<listing-id-4>",
        "street": "<street>",
        "query": ("Looking for a 3-bedroom freehold house with a garage, within "
                  "30 minutes of Clapham Junction, budget around 500k (450k to "
                  "550k). No specific area."),
    },
    {
        # floorplan VLM feature: total floor area. "Spacious" expressed as a
        # min_total_sqft floorplan constraint — unique after commute.
        "name": "spacious-3bed-canarywharf",
        "target_id": "<listing-id-5>",
        "street": "<street>",
        "query": ("I need a really spacious 3-bedroom — at least 1,800 sq ft of "
                  "internal floor area — within 20 minutes of Canary Wharf, "
                  "budget around 750k, up to 800k."),
    },
    {
        # decor as the SOLE discriminator: 8 similar 2-bed flats <=15 min of
        # Liverpool Street under 900k; the target is the highest-decor one. Forces
        # get_decor / screen_decor_value ranking, not a portal filter.
        "name": "decor-pick-liverpoolst",
        "target_id": "<listing-id-6>",
        "street": "<street>",
        "query": ("Find me the best-presented 2-bed flat within 15 minutes of "
                  "Liverpool Street, budget up to 900k - interior finish and "
                  "decor quality matter most to me."),
    },
    {
        # floorplan VLM: garden + 3 reception rooms, in a West-London hub
        # (Ealing Broadway) not covered by the others - tests commute breadth.
        "name": "garden-receptions-ealing",
        "target_id": "<listing-id-7>",
        "street": "<street>",
        "query": ("A 4-bedroom freehold house with a garden and at least 3 "
                  "reception rooms, within 25 minutes of Ealing Broadway, budget "
                  "around 1.3 million (1.2m to 1.4m)."),
    },
]


def _postcode_where(pp, col_norm):
    """Mirror the tools' outcode-exact-OR-prefix postcode match. Returns
    (clause, params) or (None, [])."""
    oc = "".join(str(pp).split()).upper()
    if not oc:
        return None, []
    if oc.isalpha():
        # area prefix ("SW") — require a digit after the letters (boundary)
        return (f"(UPPER({col_norm}) LIKE ? AND substr(UPPER({col_norm}),?,1) BETWEEN '0' AND '9')",
                [oc + "%", len(oc) + 1])
    # full/partial outcode — exact outcode OR prefix (mirrors search_floorplans)
    return (f"(UPPER(substr({col_norm},1,length({col_norm})-3)) = ? OR UPPER(substr({col_norm},1,?)) = ?)",
            [oc, len(oc), oc])


def _type_where(pt, col):
    p = str(pt).strip().lower()
    if p in ("house", "houses"):
        return f"LOWER({col}) IN ({','.join('?'*len(HOUSE))})", list(HOUSE)
    if p in ("flat", "flats", "apartment", "apartments"):
        return f"LOWER({col}) IN ({','.join('?'*len(FLAT))})", list(FLAT)
    return f"LOWER({col}) LIKE ?", ["%" + p + "%"]


def _commute_where(a, postcode_col):
    """Mirror the search tools' commute catchment filter. Returns (clause, params)."""
    hub = (a.get("commute_hub") or "").strip()
    cmax = a.get("commute_max_minutes")
    if not hub or cmax is None:
        return None, []
    npc = f"replace({postcode_col},' ','')"
    sec = f"(UPPER(substr({npc},1,length({npc})-3)) || ' ' || UPPER(substr({npc},length({npc})-2,1)))"
    return (f"{sec} IN (SELECT sector FROM sector_hub_commute WHERE LOWER(hub)=LOWER(?) "
            f"AND minutes IS NOT NULL AND minutes <= ?)", [hub, int(cmax)])


def replay(tool: str, a: dict, target_id: str):
    """Replay one search call against the DB, mirroring the tool's WHERE.
    Returns (target_in_results, result_count)."""
    norm = "replace(postcode,' ','')"
    if tool == "search_properties":
        where = ["delisted_date IS NULL", "(canonical_id IS NULL OR canonical_id=id)"]
        params = []
        if a.get("bedrooms_min") is not None:
            where.append("bedrooms >= ?"); params.append(int(a["bedrooms_min"]))
        if a.get("bedrooms_max") is not None:
            where.append("bedrooms <= ?"); params.append(int(a["bedrooms_max"]))
        if a.get("price_min") is not None:
            where.append("asking_price >= ?"); params.append(int(a["price_min"]))
        if a.get("price_max") is not None:
            where.append("asking_price <= ?"); params.append(int(a["price_max"]))
        if a.get("bathrooms_min") is not None:
            where.append("bathrooms >= ?"); params.append(int(a["bathrooms_min"]))
        if a.get("tenure"):
            where.append("UPPER(tenure) LIKE ?"); params.append("%" + str(a["tenure"]).upper() + "%")
        if a.get("min_epc_efficiency") is not None:
            where.append("epc_energy_efficiency IS NOT NULL AND epc_energy_efficiency >= ?")
            params.append(int(a["min_epc_efficiency"]))
        if a.get("property_type"):
            c, p = _type_where(a["property_type"], "property_type"); where.append(c); params += p
        if a.get("postcode_prefix"):
            c, p = _postcode_where(a["postcode_prefix"], f"UPPER({norm})" if str(a["postcode_prefix"]).strip().isalpha() else norm)
            if c: where.append(c); params += p
        cc, cp = _commute_where(a, "postcode")
        if cc: where.append(cc); params += cp
        sql = f"SELECT id FROM rm_sales_overview WHERE {' AND '.join(where)}"

    elif tool == "search_floorplans":
        where = ["o.delisted_date IS NULL", "(o.canonical_id IS NULL OR o.canonical_id=o.id)", "f.ok=1"]
        params = []
        if a.get("bedrooms_min") is not None:
            where.append("o.bedrooms >= ?"); params.append(int(a["bedrooms_min"]))
        if a.get("bedrooms_max") is not None:
            where.append("o.bedrooms <= ?"); params.append(int(a["bedrooms_max"]))
        if a.get("price_min") is not None:
            where.append("o.asking_price >= ?"); params.append(int(a["price_min"]))
        if a.get("price_max") is not None:
            where.append("o.asking_price <= ?"); params.append(int(a["price_max"]))
        if a.get("bathrooms_min") is not None:
            _n = int(a["bathrooms_min"])
            where.append("(f.bathrooms >= ? OR o.bathrooms >= ?)"); params.extend([_n, _n])
        if a.get("min_reception_rooms") is not None:
            where.append("f.reception_rooms >= ?"); params.append(int(a["min_reception_rooms"]))
        if a.get("min_total_sqft") is not None:
            where.append("f.total_sqft >= ?"); params.append(float(a["min_total_sqft"]))
        if a.get("has_garden"):
            where.append("f.has_garden = 1")
        if a.get("has_garage"):
            where.append("f.has_garage = 1")
        if a.get("property_type"):
            c, p = _type_where(a["property_type"], "o.property_type"); where.append(c); params += p
        if a.get("postcode_prefix"):
            colnorm = "UPPER(replace(o.postcode,' ',''))" if str(a["postcode_prefix"]).strip().isalpha() else "replace(o.postcode,' ','')"
            c, p = _postcode_where(a["postcode_prefix"], colnorm)
            if c: where.append(c); params += p
        cc, cp = _commute_where(a, "o.postcode")
        if cc: where.append(cc); params += cp
        sql = (f"SELECT o.id FROM rm_sales_overview o "
               f"JOIN floorplan_vlm_results f ON f.rm_uuid=o.id "
               f"AND f.processed_at=(SELECT MAX(f2.processed_at) FROM floorplan_vlm_results f2 WHERE f2.rm_uuid=o.id) "
               f"WHERE {' AND '.join(where)}")
    else:
        return None, None

    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        ids = {str(r[0]) for r in db.execute(sql, params).fetchall()}
    finally:
        db.close()
    return target_id in ids, len(ids)


def main():
    admin_key = _admin_key()
    out = []
    for nd in NEEDLES:
        print(f"\n=== {nd['name']} (target {nd['target_id']}) ===")
        print(f"  Q: {nd['query']}")
        res = run_case({"query": nd["query"], "lang": "en"}, admin_key, timeout=300)

        recall_hit, replay_detail = False, []
        for tc in res["tool_calls"]:
            if tc["name"] not in ("search_properties", "search_floorplans"):
                continue
            hit, n = replay(tc["name"], tc["args"], nd["target_id"])
            if hit is None:
                continue
            replay_detail.append({"tool": tc["name"], "args": tc["args"],
                                  "result_count": n, "target_in_results": hit})
            recall_hit = recall_hit or hit

        ans = res["answer"]
        # EXACT surfacing = the target's own listing id (rides in its URLs).
        # The street name is only a SOFT signal: in dense inventory (same
        # building, same road) it false-positives on a sibling listing, so we
        # report it separately and never let it count as an exact hit.
        exact = nd["target_id"] in ans
        street_soft = nd["street"].lower() in ans.lower()

        for tc in res["tool_calls"]:
            arg_s = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
            print(f"  -> {tc['name']}({arg_s})")
        for rd in replay_detail:
            print(f"     replay {rd['tool']}: {rd['result_count']} results, target_in={rd['target_in_results']}")
        print(f"  RECALL (a search of the agent's can find it): {recall_hit}")
        print(f"  EXACT SURFACED (target id in answer): {exact}   (street-soft={street_soft})")
        print(f"  {len(ans)} chars, {res['elapsed_s']}s, errors={res['errors']}")

        out.append({
            "name": nd["name"], "target_id": nd["target_id"], "query": nd["query"],
            "tool_calls": res["tool_calls"], "replay": replay_detail,
            "recall_hit": recall_hit, "exact_surfaced": exact, "street_soft": street_soft,
            "answer": ans, "elapsed_s": res["elapsed_s"], "errors": res["errors"],
        })

    nrecall = sum(1 for o in out if o["recall_hit"])
    nexact = sum(1 for o in out if o["exact_surfaced"])
    print(f"\n{'='*50}")
    print(f"RECALL (a search could find it):     {nrecall}/{len(out)}")
    print(f"EXACT SURFACED (target id in answer): {nexact}/{len(out)}")
    Path("/tmp/needle_results.json").write_text(json.dumps(out, indent=2))
    print("results -> /tmp/needle_results.json")


if __name__ == "__main__":
    main()
