"""Shared outdoor-space classification — the single source three call sites used to copy.

The floorplan VLM's `room_type` is unreliable for outdoor spaces: ~11% of rows typed
'balcony'/'terrace' are actually ground-level patios / gardens / courtyards / decking
(the model transcribes the drawn box but mis-types the space). The *label* text
("Patio 28'3\" …") is the trustworthy signal.

Two consumers, one classification:
  - radar balcony match / chat search want ELEVATED private outdoor only, so they
    exclude ground labels  → `ground_label_exclusion_sql()`.
  - the daily-picks outdoor board keeps every private outdoor space but must name it
    HONESTLY (a patio is not a balcony) → `outdoor_display_type()`.

radar_vocab.py, search_properties.py and daily_picks.py each previously inlined this
logic; picks drifted and never got the guard, so a ground patio surfaced on the board
labelled a terrace. Keep the logic here so they cannot diverge again.
"""

# Ground-level outdoor labels the VLM often mis-types as balcony/terrace. Substring
# matched, so "deck" also catches "decking". 'garden' is handled separately below
# because of the 'roof garden' carve-out (a roof garden IS elevated).
GROUND_OUTDOOR_KEYWORDS = ("patio", "courtyard", "deck", "pergola")


def outdoor_display_type(room_type, label):
    """Honest outdoor-space type for display, derived from the floorplan LABEL first
    (room_type is unreliable — see module docstring), falling back to the coarse
    room_type only when the label carries no recognised outdoor keyword. Ground
    patios/gardens keep their true name instead of masquerading as a balcony/terrace."""
    l = (label or "").lower()
    if "roof garden" in l or "roof terrace" in l:   # elevated → normalise spelling
        return "roof terrace"
    if "patio" in l:
        return "patio"
    if "courtyard" in l:
        return "courtyard"
    if "decking" in l or "deck " in l:
        return "decking"
    if "veranda" in l:
        return "veranda"
    if "garden" in l:            # ground garden (roof garden already returned above)
        return "garden"
    if "balcon" in l:
        return "balcony"
    if "terrace" in l:
        return "terrace"
    rt = room_type or "outdoor"  # label unhelpful → trust room_type, fixing the enum spelling
    return "roof terrace" if rt == "roof_terrace" else rt


def is_ground_level_label(label):
    """True when the label names a ground-level outdoor space (patio/courtyard/decking/
    pergola/ground garden) — but NOT 'roof garden' (elevated). Excludes ground spaces
    from an elevated-'balcony' request."""
    l = (label or "").lower()
    if "roof garden" in l:
        return False
    if "garden" in l:
        return True
    return any(k in l for k in GROUND_OUTDOOR_KEYWORDS)


def ground_label_exclusion_sql(label_expr):
    """SQL fragment (leading space, no bind params) excluding ground-level outdoor
    labels from an elevated-balcony match, preserving the 'roof garden' carve-out.
    `label_expr` is the label column expression (e.g. "ra.label"). This is the exact
    set of NOT-LIKE clauses radar_vocab / search_properties historically inlined —
    now defined once so all elevated-outdoor predicates stay in lockstep."""
    low = f"LOWER(COALESCE({label_expr},''))"
    parts = [f" AND {low} NOT LIKE '%{k}%'" for k in GROUND_OUTDOOR_KEYWORDS]
    parts.append(f" AND ({low} NOT LIKE '%garden%' OR {low} LIKE '%roof garden%')")
    return "".join(parts)
