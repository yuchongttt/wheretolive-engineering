# Benchmark results — gold set vs local Qwen baselines (May 2026)

**What this is.** Aggregate numbers from the offline benchmark in `benchmark/`: a gold
reference set (used only to score answers) built by Claude (Opus author, Sonnet critic) and a series of local
Qwen3.5 baselines, all scored by the same LLM critic. The runs happened on
**2026-05-09 .. 2026-05-11** (the `ran_at` / `generated_at` / `scored_at` timestamps inside
the files). The tables were computed on 2026-09-30 by
[`benchmark/summarize_results.py`](benchmark/summarize_results.py) over the private repo's
`data/benchmark/*.jsonl` (commit `c7c3b4ae`). The raw files are not published: tool results
inside them carry listing URLs and addresses.

Reproduce (on a checkout that has the data):

```bash
D=data/benchmark
python3 benchmark/summarize_results.py $D                                   # all files
python3 benchmark/summarize_results.py $D --only-ids-of $D/qwen_baseline_full_9b_off.jsonl \
    --group-by $D/queries_filtered.jsonl:filter_class \
    --detail-regex '585|788|995|Canning Town|52\.5|40\.8|4\.6|1\.96|South Quay'  # tables 1-3
python3 benchmark/summarize_results.py $D --group-by $D/queries_filtered.v2.jsonl:split  # table 4
```

## What was run

| run | file | model / setup | queries |
|---|---|---|---|
| gold | `gold_answers.jsonl` + `.critic.jsonl` | Claude Opus + 2 MCP tools (`search_properties`, `get_postcode_scores`), Claude Sonnet critic, up to 2 revisions | 210 |
| no-tools | `qwen_baseline_full_9b_off*.jsonl` | Qwen3.5-9B via Ollama, thinking off, no tools | 50 |
| tools v0..v4 | `qwen_baseline_full_9b_off_mcp_v{0..4}*.jsonl` | Qwen3.5-9B, thinking off, same 2 tools executed in Python, successive prompt/decoding iterations | 50 |
| thinking A/B | `qwen_thinking_compare{,_4b}.jsonl` | Qwen3.5 9B and 4B, thinking on vs off, 3 queries each, no critic | 3 |

The 50 queries are hand-written (`seed_curated_queries.py`), in Chinese (the product is
bilingual EN/ZH): 7 filter, 17 advice, 1 compare, 25 follow-up ("is *this* listing
overpriced?"). Follow-ups get a synthetic turn-1 context describing an anchor listing or
postcode. The 210-query v2 set (`expand_curated_queries.py`) adds anchor ids and a split:
train 171 / test_query 29 / test_anchor 10 (held-out anchors). `…_mcp.jsonl` is byte-identical
to `…_mcp_v4.jsonl` (the summariser reports this), i.e. v4 is the last run of
`run_qwen_9b_off_mcp.py`.

## 1. Critic verdicts on the 50 original queries

"Parse failures" are critic replies that were not valid JSON; `parse_critic()` records them as
`needs_revision` with a single low-severity marker. *Pass rate (judged)* excludes them from
the denominator.

| run | scored | pass | needs_revision | parse failures | pass rate | pass rate (judged) | cases with a high issue | cases with tool_hallucination |
|---|---|---|---|---|---|---|---|---|
| gold (50 of 210) | 2026-05-09 | 39 | 11 | 3 | 0.78 | 0.83 | 4 | 8 |
| no-tools | 2026-05-09 | 25 | 25 | 8 | 0.50 | 0.595 | 13 | 1 |
| tools v0 | 2026-05-09 | 1 | 49 | 6 | 0.02 | 0.023 | 41 | 29 |
| tools v1 | 2026-05-10 | 5 | 45 | 10 | 0.10 | 0.125 | 31 | 24 |
| tools v2 | 2026-05-10 | 7 | 43 | 10 | 0.14 | 0.175 | 31 | 21 |
| tools v3 | 2026-05-10 | 3 | 47 | 6 | 0.06 | 0.068 | 38 | 31 |
| tools v4 | 2026-05-10 | 12 | 38 | 9 | 0.24 | 0.293 | 26 | 23 |

Whole gold set (210): 160 pass / 50 needs_revision, 17 parse failures, pass rate 0.762
(judged 0.829), 14 cases with a high issue, 31 with a tool_hallucination.

## 2. Issues by severity and category (50 queries; parse-failure markers excluded)

| run | high | medium | low | tool_halluc. | general_halluc. | missing_hedge | off_domain | format | incomplete |
|---|---|---|---|---|---|---|---|---|---|
| gold | 5 | 30 | 37 | 11 | 12 | 5 | 0 | 34 | 10 |
| no-tools | 19 | 28 | 22 | 1 | 46 | 11 | 0 | 5 | 6 |
| tools v0 | 87 | 52 | 16 | 54 | 45 | 16 | 0 | 19 | 21 |
| tools v1 | 54 | 52 | 11 | 39 | 31 | 16 | 0 | 7 | 24 |
| tools v2 | 62 | 44 | 9 | 38 | 34 | 11 | 0 | 10 | 22 |
| tools v3 | 89 | 50 | 13 | 67 | 26 | 16 | 0 | 16 | 27 |
| tools v4 | 46 | 46 | 15 | 42 | 24 | 11 | 0 | 9 | 21 |

Whole gold set (210): high 15 / medium 130 / low 155; tool_hallucination 37,
general_hallucination 64, missing_hedge 32, off_domain 0, format 144, incomplete 23.

## 3. Follow-up queries vs the rest (50 queries)

The last column counts tool_hallucination issues whose text quotes one of the figures the
injected turn-1 context supplied (regex in the command above) — see finding 5.

| run | follow-up: pass / 25 | other: pass / 25 | follow-up tool_halluc. issues | … quoting a context figure |
|---|---|---|---|---|
| gold | 16 | 23 | 8 | 2 |
| no-tools | 18 | 7 | 1 | 0 |
| tools v0 | 0 | 1 | 36 | 22 |
| tools v1 | 1 | 4 | 31 | 20 |
| tools v2 | 1 | 6 | 33 | 17 |
| tools v3 | 0 | 3 | 43 | 25 |
| tools v4 | 2 | 10 | 31 | 19 |

## 4. Gold set by split (210 queries)

| split | cases | pass | parse failures | tool_halluc. issues |
|---|---|---|---|---|
| train | 171 | 129 | 13 | 30 |
| test_query (new phrasings) | 29 | 24 | 3 | 4 |
| test_anchor (held-out anchors) | 10 | 7 | 1 | 3 |

Gold pipeline status (`gold_answers.jsonl`): 168 `ok`, 42 `needs_human_review`; critic rounds
per case 1: 108, 2: 65, 3: 37.

## 5. Run characteristics

| run | ran | ok | errors / empty answers | tool calls per case | zero-tool cases | tool results stored as | mean wall s | mean answer chars |
|---|---|---|---|---|---|---|---|---|
| no-tools | 05-09 | 50 | 0 / 0 | – | – | – | 22.3 | 531 |
| tools v0 | 05-09 | 49 | 1 (HTTP 500) / 1 | 2.6 | 7 | 300-char previews | 60.2 | 831 |
| tools v1 | 05-10 | 50 | 0 / 0 | 3.0 | 4 | full | 65.4 | 628 |
| tools v2 | 05-10 | 49 | 1 (max tool rounds) / 1 | 3.8 | 1 | full | 81.2 | 874 |
| tools v3 | 05-10 | 48 | 0 / 2 | 2.2 | 5 | full | 76.0 | 1029 |
| tools v4 | 05-10 | 50 | 0 / 0 | 2.6 | 1 | full | 56.1 | 551 |
| gold (210) | 05-09..10 | – | 0 / 0 | 2.3 | 55 | full, capped at 4,000 chars | – | 1154 |

Thinking A/B (3 queries per cell, no critic scoring):

| model | thinking | mean wall s | mean eval tokens | mean answer chars | empty answers |
|---|---|---|---|---|---|
| qwen3.5:9b | off | 21.6 | 254 | 530 | 0 |
| qwen3.5:9b | on | 161.0 | 1941 | 402 | 0 |
| qwen3.5:4b | off | 20.0 | 338 | 669 | 0 |
| qwen3.5:4b | on | 184.6 | 3079 | 209 | 1 (hit the 4,096-token cap while thinking) |

A separate 9B thinking-on attempt (`qwen_baseline_small.jsonl`, 1 case) timed out at 240 s.
All Qwen baselines above therefore ran with thinking off. (`run_qwen_baseline.py` records the
counter-argument observed the same day: with thinking off, 9B misidentified what E14 is.)

## How to read these numbers — findings and caveats

1. **One run each, no repeats.** Every row is one sample of a stochastic model scored by one
   sample of a stochastic critic. Differences of a few cases out of 50 are not evidence.
2. **Critic parse failures are 12–20% of each 50-query file** (6–10 cases) and 8% of the gold
   set (17/210). They are critic noise, not answer quality; that is why the judged pass rate
   exists. (Before a 2026-05-09 fix they were recorded as *high*-severity issues.)
3. **The critic changed while these runs were being scored.** The 2026-05-09/10 calibration
   work raised the per-tool-result window in the critic prompt from 1,200 to 5,000 chars and
   made `critic_score.py` re-execute tools whose stored result looked truncated (v0 stored
   only 300-char previews). Re-execution hits the database *at scoring time*, which is not
   necessarily what the model saw. The files do not record which critic version scored them.
4. **Gold is not an independent reference.** Gold answers were revised against this same
   critic until it passed (or two revisions ran out), so its pass rate is optimistic by
   construction. Use it as "what the pipeline can reach", not as a neutral upper bound.
5. **Critic blind spot — conversation context.** Follow-up queries are answered with a
   synthetic turn-1 context (anchor listing: 2-bed, 788 sq ft, £585,000, 995-year lease;
   anchor postcode: Price 52.5, Schools 40.8, …), but `make_critic_prompt()` only receives the
   bare query, the tool calls and the answer. The critic therefore flags facts taken from the
   context as fabricated: in v4, 19 of the 31 tool_hallucination issues on follow-ups quote a
   context figure (case 3 below). This is a lower bound (the regex only matches numbers and
   one place name). Follow-up rows in tables 1–3 overstate failure until the critic is given
   the same context.
6. **Critic blind spot — long tool results.** The critic sees at most the first 5,000 chars of
   each tool result. `get_postcode_scores` returns 8.9–12.8k chars in the stored v4/v1 results,
   with its repeat-sales / CAGR block starting around char 6,550; the gold recorder also caps
   stored results at 4,000 chars (54 of 487 gold tool results hit that cap). A CAGR figure
   read correctly from the tool can only be judged "not in the tool result". Few flags cite
   CAGR / repeat sales (gold 3, v0–v4: 4 / 2 / 3 / 9 / 1), but they are unverifiable by design.
7. **"No tools 0.50 vs tools 0.24" is not like-for-like.** `run_qwen_9b_off.py` injects the
   turn-1 context only when `intent_subtype == "property"` / `"area"`, but the query set uses
   `property_eval` / `area_eval`, so the no-tools run never saw it. Its follow-up answers ask
   the user to share the listing — and 18/25 of those pass. On the other 25 queries the scores
   are 7 (no tools) vs 10 (tools v4). The failure modes also differ: without tools the critic
   can only find general hallucinations (46 issues); with tools it mostly finds misread tool
   output (42 tool_hallucination issues in v4).
8. **Prompt examples leak into tool arguments.** `AUTHOR_SYSTEM` shows a citation example with
   postcode E14 0GQ. Qwen called `get_postcode_scores("E14 0GQ")` on 7 / 2 / 9 / 0 / 9 of the 25
   non-follow-up queries in v0 / v1 / v2 / v3 / v4 — queries that never mention it.
9. **Few-shot demos leak into answers.** v3 is the only file where non-follow-up answers
   reproduce figures from the few-shot demo block (6 answers). It also has the most
   tool_hallucination issues (67). This matches the runner's own note that the few-shot batch
   "encouraged MORE cite invention" and was dropped in the next batch. The files do not record
   which prompt variant produced them; the v0–v4 ↔ batch mapping is inferred from timestamps
   and this evidence.
10. **The critic is fallible in its own right.** Examples: case 4's critic aside places the
    Piccadilly line at Harrow-on-the-Hill (it is the Metropolitan line); in gold `curated_19`,
    round 0 claims IMD fields are absent from a tool result that contains `"imd_decile": 1`.
    Flags are prompts for a human to read the transcript, not verdicts.

**Bottom line.** On identical queries and the same critic, the tool-using 9B
baseline passed 12/50 against 39/50 for the gold pipeline, and its dominant failure was
misattributing tool output. Findings 5–7 mean the follow-up rows need a critic fix before they
can rank models; the non-follow-up rows (table 3, "other") are the cleaner comparison.

## Illustrative cases

Queries are translated from Chinese; listing ids, URLs, street and building names are
redacted. Tool names and arguments are verbatim.

### Case 1 — caught: tool output misread, and a silent tool error (tools v4, `curated_20`, filter)

**Query:** "Which areas are best for buying a Victorian house?"

**Tool calls**
1. `search_properties(bedrooms_min=2, bedrooms_max=4, price_max=1500000, limit=25)` → 25 listings, most of them flats
2. `search_properties(bedrooms_min=2, bedrooms_max=4, price_max="1.5e+06", property_type="Terraced", limit=25)` →
   `Error executing search_properties: invalid literal for int() with base 10: '1.5e+06'`

**Answer (excerpt):** "Walthamstow (E17): a 2-bed Victorian terraced house for sale at
£575,000 … Greenwich: a 2-bed terraced house at £475,000 by the Thames … Fulham (SW6): a 2-bed
terraced house at £750,000, in Kensington and Chelsea … Ealing (W5): a 2-bed terraced house at
£500,000 …" The failed second search is not mentioned.

**Critic → needs_revision:** 3 × high tool_hallucination (the Greenwich, Fulham and Ealing
listings are *Flats* in the tool result but presented as terraced houses); high
general_hallucination (Fulham is in Hammersmith & Fulham, not Kensington & Chelsea); medium
general_hallucination (Seven Kings described by the wrong station); medium incomplete (the tool
result did contain real terraced houses that were ignored).

**What it shows:** field-level misattribution that string checks cannot see. The critic did
not flag the swallowed tool error or the float-as-string argument — that belongs to
trace-level checks and to argument coercion in the tool layer (`../tools/`).

### Case 2 — pass: a "no data" tool result respected (tools v4, `curated_07`, advice)

**Query:** "Which areas are best for buy-to-let investment?"

**Tool calls:** `search_properties(bedrooms_min=1, bedrooms_max=2, price_min=200000, price_max=400000, limit=25)`
→ 25 listings; `get_postcode_scores(postcode="N4")`, `("KT9")`, `("E17")` → each
"Postcode …: not yet evaluated … Do NOT make up scores."

**Answer (excerpt):** "The tools have not returned scores for these postcodes, so I can't
quote exact figures for N4, KT9 or E17. From the listings the search did return … East London
(e.g. Walthamstow E17): the tool shows a 1-bed flat at £205,000 … Next step: evaluate a
specific postcode on wheretolive.xyz."

**Critic → pass**, one low format issue (prices quoted without the listing ids that would make
them checkable). This is the Qwen prompt's first hard rule ("a no-data tool result is a real
signal, never back-fill a score") working as intended.

### Case 3 — critic false positive: context the critic never saw (tools v4, `curated_22`, follow-up)

**Query:** "Is this listing overpriced?" — asked as turn 2 after a synthetic turn 1 describing
the anchor listing (2-bed, 788 sq ft, £585,000, Poplar E14).

**Tool calls:** `search_properties(bedrooms_min=2, bedrooms_max=2, price_max=650000, postcode_prefix="E14", limit=10)`;
`get_postcode_scores(postcode="E14 <inward code>")` (the anchor's full postcode, redacted)
→ not yet evaluated.

**Answer (excerpt):** "2-bed flats in E14 in the tool results range from £500,000 to
£650,000 … the listing you mentioned (788 sq ft, 2-bed) at £585,000 … about £742/sq ft, against
£502/sq ft for a 995 sq ft flat at £500k in the same outcode … on the high side, but not
clearly overpriced."

**Critic → needs_revision:** 2 × high tool_hallucination ("asserts a specific target listing
… at £585,000 … that does not appear in any tool result, making this a fabricated property";
"£742/sq ft is derived from the fabricated listing") and a high incomplete ("the query gives
no listing URL or id; the agent should have asked which listing").

**Why that is wrong:** the price and size came from the context the model was given; the
critic prompt only contains the bare query. The answer is grounded. See finding 5.

### Case 4 — no tools: geography hallucinations, and a fallible critic (no-tools, `curated_01`, filter)

**Query:** "With a £700k budget, which London areas are good for buying a 2-bed?"

**Tool calls:** none (baseline without tools).

**Answer (excerpt):** "Harrow … close to Heathrow airport … Enfield: close to Heathrow, direct
Overground … Barnet (Hatch End / Finchley) … Ealing … a top pick in North London."

**Critic → needs_revision:** high — Enfield is in North London (EN postcodes), about 30 km
from Heathrow (TW6); high — Ealing is West London; medium — Hatch End (HA5) is in the London
Borough of Harrow, not Barnet; medium — Harrow is not next to Heathrow; medium missing_hedge —
price bands with no source or caveat.

**Note:** inside the Harrow item the critic adds that the Piccadilly line runs through
Harrow-on-the-Hill. That aside is itself wrong (Metropolitan line) — the finding stands, the
reasoning around it is not ground truth.

### Case 5 — the gold loop: author → critic → revise (gold, `curated_09`, advice)

**Query:** "Which London areas have the greatest future price-growth potential?"

**Tool calls (Claude Opus author):** `get_postcode_scores` for `E16 2QU`, `E20 1FA`,
`SE10 0SQ`, `IG11 7RY` → three "not yet evaluated", one full result for E20 1FA (stored
truncated at 4,000 chars).

**Round 0 critic → needs_revision:** high tool_hallucination — 10-year / 5-year CAGR figures
tagged `[postcode=E20 1FA]` are not in the tool result; high — repeat-sale transactions in a
named building cited as tool data; medium — two school names tagged as tool-sourced, while the
tool only returned the aggregate schools score.

**Round 1 (after revision) → pass** (one low incomplete: commute details not surfaced).
**Final answer (excerpt):** "Data-backed sample — E20 (Stratford / Olympic Park): Price
73.2/100, Schools 56.4/100 [postcode=E20 1FA] … specific annual growth, repeat sales and school
lists were not returned by the tool this time … Other candidates from general knowledge —
industry views, not tool data, verify them yourself: Royal Docks (E16) …"

**Caveat:** by finding 6, round 0 could not have verified CAGR figures either way; the
revision made the answer safer, and possibly also removed figures that were real.
