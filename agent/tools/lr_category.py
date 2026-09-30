"""Shared LR category vocabulary for chat-skills (R3-1-refix).

One truth source for the category-B taxonomy that R3-1 spread across six
hand-written copies (with wording drift in fetch_listing and a NULL-semantics
split between `== 'A'` and `!= 'B'` predicates — review findings 11/14).

Conventions pinned here:
  * standard  = category exactly 'A'. NULL/blank is NOT standard — an
    unrecorded category must never enter a "standard open-market sales only"
    statistic — but it is also NOT labelled category B (that would assert
    repossession/BTL/corporate about a sale that is merely unrecorded).
  * Labels are the exact strings tools append to row output; keep them
    byte-identical across tools so the model sees ONE flag taxonomy.
"""

# What category B actually is — HMLR's "additional price paid" entries.
CATEGORY_B_GLOSS = ("repossessions, mortgage-identifiable buy-to-lets, "
                    "corporate transfers")

CATEGORY_B_LABEL = " · ⚠ LR category B — not a standard open-market sale"
CATEGORY_UNRECORDED_LABEL = (" · ⚠ LR category unrecorded — "
                             "not verifiably a standard sale")

# LR property_type codes.
LR_TYPE = {"F": "Flat", "T": "Terraced", "S": "Semi-detached",
           "D": "Detached", "O": "Other"}


def is_category_b(category) -> bool:
    return (category or "").strip().upper() == "B"


def is_standard(category) -> bool:
    """Standard open-market sale: category exactly 'A' (NULL is not)."""
    return (category or "").strip().upper() == "A"


def category_label(category) -> str:
    """Row-label suffix for a single leg/row; '' for a standard sale."""
    if is_standard(category):
        return ""
    if is_category_b(category):
        return CATEGORY_B_LABEL
    return CATEGORY_UNRECORDED_LABEL
