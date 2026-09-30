#!/usr/bin/env python3
"""Expand curated query set from 50 → 210 (add 160 new) with anchor & split tagging.

For follow_up queries (property_eval + area_eval) we attach a real
property listing or evaluated postcode as the synthetic prior-turn
anchor. Anchors are picked from a small registry of 4 properties +
4 areas (real DB rows). Each query is constrained to anchors that
make sense (e.g. "ground rent" only applies to leasehold flats).
(Public copy: property-anchor street names / full postcodes are redacted
in the labels below; the labels are informational only.)

Mixed train/test split:
  train               – 70% — uses anchors A1/A2/A3 + Z1/Z2/Z3
  test_query_general  – ~17% — same anchor pool as train, NEW query patterns
  test_anchor_general – ~13% — hold-out anchors A4 (Period House) + Z4 (St Paul's)
                        only — tests anchor generalization

Non-follow_up queries (advice / filter / compare) don't need anchors
but still get a split label.

Output schema additions per record:
  anchor_id  : str | None  ("A1"–"A4" / "Z1"–"Z4" / null for non-followup)
  split      : "train" | "test_query" | "test_anchor"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Root of the production checkout (holds data/benchmark/).
REPO = Path(os.environ.get("WTL_ROOT", Path(__file__).resolve().parents[3]))
DATA = REPO / "data" / "benchmark"

# ──────────────────────────────────────────────────────────────────────
# ANCHOR REGISTRY — must stay in sync with build_gold_answers.ANCHORS
# ──────────────────────────────────────────────────────────────────────

PROPERTY_ANCHORS = {
    "A1": {
        "tags": {"leasehold", "flat", "modern", "mid_price", "docklands", "multi_bed", "shared_freehold"},
        "label": "<development> · Poplar E14 · £585k · 2-bed Flat · 995-yr leasehold (Docklands, modern)",
    },
    "A2": {
        "tags": {"low_price", "first_time_buyer", "maisonette", "south_east", "single_bed", "older"},
        "label": "<street> · Ladywell SE13 · £300k · 1-bed Maisonette (Lewisham, FTB)",
    },
    "A3": {
        "tags": {"leasehold", "flat", "modern", "mid_price", "docklands", "multi_bed", "shared_freehold"},
        "label": "<street> · E14 · £635k · 2-bed Flat · leasehold (Docklands, modern)",
    },
    "A4": {
        "tags": {"freehold", "house", "period", "high_price", "west_london", "multi_bed", "extendable", "older"},
        "label": "<street> · W6 · £1,000,000 · 3-bed Terraced (Hammersmith, Period)",
    },
}

AREA_ANCHORS = {
    "Z1": {"tags": {"evaluated", "urban", "docklands", "east"}, "label": "E14 9NA · Isle of Dogs / Canary Wharf"},
    "Z2": {"tags": {"evaluated", "urban", "central", "north", "gentrified"}, "label": "N1 9AA · Islington (King's Cross side)"},
    "Z3": {"tags": {"evaluated", "residential", "family_friendly", "north"}, "label": "N5 1FL · Highbury"},
    "Z4": {"tags": {"evaluated", "urban", "central"}, "label": "EC4M 8AD · St Paul's"},
}

# A4 + Z4 are HOLD-OUT for test_anchor_general. Train + test_query never use these.
TRAIN_PROPERTY_ANCHORS = ["A1", "A2", "A3"]
HOLDOUT_PROPERTY_ANCHORS = ["A4"]
TRAIN_AREA_ANCHORS = ["Z1", "Z2", "Z3"]
HOLDOUT_AREA_ANCHORS = ["Z4"]


# ──────────────────────────────────────────────────────────────────────
# 160 NEW QUERIES — (text, class, subtype, required_tags_per_query)
# required_tags is list of sets; query is compatible with an anchor if
# the anchor's tag set is a SUPERSET of at least one required set.
# Empty list = compatible with everything.
# ──────────────────────────────────────────────────────────────────────

NEW_QUERIES: list[tuple[str, str, str, list[set[str]]]] = [
    # ───── filter / area_search ─────
    ("£500k 在 London 哪些区适合一居室？", "filter", "area_search", []),
    ("£900k 想买 3 居 House，哪些区域可选？", "filter", "area_search", []),
    ("£300k 以内 London 哪里能买到一居室？", "filter", "area_search", []),
    ("£1.2M 预算 4 居 House 推荐哪些区域？", "filter", "area_search", []),
    ("£450k 想买带花园的 2 居有哪些区？", "filter", "area_search", []),
    ("£600k 想买公寓 + 健身房，哪些 zone 1-2 区域有？", "filter", "area_search", []),
    ("£250k 在 London 能买什么样的房子？", "filter", "area_search", []),
    ("£800k Studio 公寓在哪些核心区可买？", "filter", "area_search", []),
    ("£750k 想买靠河边的 2 居哪些区？", "filter", "area_search", []),
    ("£550k 想买带 parking 的 2 居哪些区？", "filter", "area_search", []),
    ("£400k 一居 + 离地铁 5 分钟内的区域？", "filter", "area_search", []),
    ("£950k Period House 哪些区域有？", "filter", "area_search", []),
    ("£350k 在 zone 3 之外能买什么？", "filter", "area_search", []),
    ("£700k 在 SE 区域可以买到什么？", "filter", "area_search", []),

    # ───── filter / commute ─────
    ("哪些区域通勤 King's Cross 最方便？", "filter", "commute", []),
    ("哪些区域适合在 Waterloo 上班？", "filter", "commute", []),
    ("从 Bank 通勤 30 分钟内的区域有哪些？", "filter", "commute", []),
    ("哪些区域适合在 Heathrow 上班？", "filter", "commute", []),
    ("哪些区域适合在 Stratford 上班？", "filter", "commute", []),
    ("哪些区域通勤 Westminster 最方便？", "filter", "commute", []),
    ("从 Liverpool Street 通勤 45 分钟内有哪些区？", "filter", "commute", []),
    ("Imperial College 附近哪些区适合教职工？", "filter", "commute", []),
    ("UCL 附近哪些区适合学生 / 教职工？", "filter", "commute", []),
    ("哪些区域适合在 Canary Wharf 上班同时又安静？", "filter", "commute", []),
    ("从 Paddington 出发 30 分钟内的居住区？", "filter", "commute", []),
    ("哪些区域跟 Victoria 通勤都方便？", "filter", "commute", []),

    # ───── filter / property_type ─────
    ("哪些区域 Edwardian house 比较多？", "filter", "property_type", []),
    ("哪些区域 Mansion Block 公寓多？", "filter", "property_type", []),
    ("哪些区域 New Build 公寓选择多？", "filter", "property_type", []),
    ("哪些区域 Mews House 多？", "filter", "property_type", []),
    ("哪些区域 Conversion Flat 多？", "filter", "property_type", []),
    ("哪些区域 Penthouse 公寓多？", "filter", "property_type", []),
    ("哪些区域 Garden Flat 比较多？", "filter", "property_type", []),
    ("哪些区域 Detached House 选择多？", "filter", "property_type", []),

    # ───── filter / yield_rank ─────
    ("哪些 outcode 短租回报最高？", "filter", "yield_rank", []),
    ("哪些区域学生租房需求最强？", "filter", "yield_rank", []),
    ("哪些区域 HMO 模式收益率最高？", "filter", "yield_rank", []),
    ("哪些区域适合 short-let（Airbnb）？", "filter", "yield_rank", []),
    ("哪些 zone 3 邮编租金回报最高？", "filter", "yield_rank", []),

    # ───── filter / zone_budget ─────
    ("Zone 3 内 £500k 能买到什么样的房？", "filter", "zone_budget", []),
    ("Zone 1 £700k 公寓在哪些区可选？", "filter", "zone_budget", []),
    ("Zone 4 £400k 能买到 2 居吗？", "filter", "zone_budget", []),
    ("Zone 5 £350k 能不能买独栋？", "filter", "zone_budget", []),
    ("Zone 2 £600k 三居可行吗？", "filter", "zone_budget", []),

    # ───── filter / schools ─────
    ("Outstanding 小学密集区有哪些？", "filter", "schools", []),
    ("Grammar School 强的区域在哪？", "filter", "schools", []),

    # ───── filter / budget_only ─────
    ("有 £600k 投资预算，London 哪些区域回报最稳？", "filter", "budget_only", []),
    ("£1M 现金，London 怎么配置房产？", "filter", "budget_only", []),

    # ───── advice / persona ─────
    # (Public copy: five queries about religion, ethnicity, sexuality or tax planning
    #  are omitted here, so this list yields 205 rather than 210 queries.)
    ("哪些区域适合刚毕业的金融从业者？", "advice", "persona", []),
    ("哪些区域适合带宠物的家庭？", "advice", "persona", []),
    ("哪些区域适合 50+ 退休夫妇？", "advice", "persona", []),
    ("哪些区域适合艺术家 / 创意工作者？", "advice", "persona", []),
    ("哪些区域适合医护人员（NHS staff）？", "advice", "persona", []),
    ("哪些区域适合在家创业 / freelancer？", "advice", "persona", []),
    ("哪些区域适合带初中生孩子的家庭？", "advice", "persona", []),

    # ───── advice / purpose ─────
    ("哪些区域适合给孩子读书买学区房？", "advice", "purpose", []),
    ("哪些区域适合给父母养老买房？", "advice", "purpose", []),
    ("哪些区域适合 flip（短期翻新转售）？", "advice", "purpose", []),
    ("哪些区域适合做 develop（扩建 / loft 加层）？", "advice", "purpose", []),
    ("哪些区域适合留学家庭的过渡居住？", "advice", "purpose", []),
    ("哪些区域适合婚后小家庭过渡？", "advice", "purpose", []),
    ("买给在 London 上学的孩子，住哪儿？", "advice", "purpose", []),
    ("买来当 holiday home 偶尔住住，选哪？", "advice", "purpose", []),

    # ───── advice / trend ─────
    ("接下来 5 年哪些区域会涨最多？", "advice", "trend", []),
    ("Crossrail 2 沿线哪些区有提前布局价值？", "advice", "trend", []),
    ("HS2 取消后哪些区被低估了？", "advice", "trend", []),
    ("Bakerloo 延伸沿线有哪些机会？", "advice", "trend", []),
    ("Royal Docks 重建对周边房价影响？", "advice", "trend", []),
    ("近 3 年哪些区域跌幅最大？", "advice", "trend", []),
    ("哪些 fringe 区域正在快速升级？", "advice", "trend", []),

    # ───── advice / lifestyle ─────
    ("哪些区域咖啡馆文化最浓？", "advice", "lifestyle", []),
    ("哪些区域绿地最多？", "advice", "lifestyle", []),
    ("哪些区域离机场近又安静？", "advice", "lifestyle", []),
    ("哪些区域生活节奏比较慢、适合 work from home？", "advice", "lifestyle", []),
    ("哪些区域跑步 / 健身设施好？", "advice", "lifestyle", []),
    ("哪些区域 farmers market 比较多？", "advice", "lifestyle", []),

    # ───── advice / general_principle ─────
    ("Freehold 一定比 Leasehold 好吗？", "advice", "general_principle", []),
    ("租期还剩多少年时不能再买？", "advice", "general_principle", []),
    ("学区房一定保值吗？", "advice", "general_principle", []),
    ("距离地铁多近算「近」？", "advice", "general_principle", []),
    ("Listed building 适不适合自住？", "advice", "general_principle", []),
    ("EPC 分数对房价影响大吗？", "advice", "general_principle", []),

    # ───── advice / general_knowledge ─────
    ("First-time buyer 有什么税务优惠？", "advice", "general_knowledge", []),
    ("买房需要多少 deposit？", "advice", "general_knowledge", []),
    ("Help to Buy 还能用吗？", "advice", "general_knowledge", []),
    ("Stamp Duty 怎么算？", "advice", "general_knowledge", []),
    ("Conveyancing 流程一般多久？", "advice", "general_knowledge", []),

    # ───── compare / type_compare ─────
    ("Flat 和 Maisonette 哪个更值得？", "compare", "type_compare", []),
    ("Terraced 和 Semi-detached 比哪个保值？", "compare", "type_compare", []),
    ("买 New build 还是二手公寓划算？", "compare", "type_compare", []),
    ("Studio 和 1 bed 投资哪个回报更好？", "compare", "type_compare", []),
    ("Period 房和现代公寓维护成本差多少？", "compare", "type_compare", []),
    ("Ground floor 和高层公寓哪个好？", "compare", "type_compare", []),
    ("自住和投资买的房一样吗？", "compare", "type_compare", []),
    ("一次性付款和按揭买，哪个划算？", "compare", "type_compare", []),
    ("Cash buyer 和 mortgage buyer 谈判优势比？", "compare", "type_compare", []),

    # ───── follow_up / property_eval (35 — anchor-tagged) ─────
    ("这个房子的物业管理怎么样？", "follow_up", "property_eval", [{"flat"}]),
    ("这个房子近期有 price reduction 吗？", "follow_up", "property_eval", []),
    ("这个房子的 ground rent 合理吗？", "follow_up", "property_eval", [{"leasehold"}]),
    ("这个房子能扩建吗？", "follow_up", "property_eval", [{"extendable"}, {"house"}]),
    ("这个 listing 的描述有没有需要警惕的措辞？", "follow_up", "property_eval", []),
    ("这个房子的卧室面积够大吗？", "follow_up", "property_eval", []),
    ("这个房子能 split rental（合租）吗？", "follow_up", "property_eval", [{"multi_bed"}]),
    ("这个 listing 的 floor plan 看起来合理吗？", "follow_up", "property_eval", []),
    ("这个房子有没有结构性问题风险？", "follow_up", "property_eval", [{"older"}, {"period"}]),
    ("这个 listing 是 chain free 吗？", "follow_up", "property_eval", []),
    ("这个房子大概什么时候能 complete？", "follow_up", "property_eval", []),
    ("这个房子可以给 buy-to-let 用吗？", "follow_up", "property_eval", []),
    ("这个房子贷款好批吗？", "follow_up", "property_eval", []),
    ("这个房子的 ceiling height 怎么样？", "follow_up", "property_eval", [{"period"}]),
    ("这个 listing 我能砍多少价？", "follow_up", "property_eval", []),
    ("这个房子的 stamp duty 大概多少？", "follow_up", "property_eval", []),
    ("这个房子的 council tax band 是多少？", "follow_up", "property_eval", []),
    ("这个房子有没有 cellar / basement？", "follow_up", "property_eval", [{"period"}, {"house"}]),
    ("这个房子值不值得请验房师？", "follow_up", "property_eval", []),
    ("这个房子的能源效率怎么样？", "follow_up", "property_eval", []),
    ("这个房子有 cladding 风险吗？", "follow_up", "property_eval", [{"flat", "modern"}]),
    ("这个房子的 lease 续到多少年了？", "follow_up", "property_eval", [{"leasehold"}]),
    ("这个房子的房东是开发商还是个人？", "follow_up", "property_eval", []),
    ("这个房子如果空置多久，房产会不会贬值？", "follow_up", "property_eval", []),
    ("这个房子的转售周期一般多久？", "follow_up", "property_eval", []),
    ("这个房子离最近的医院多远？", "follow_up", "property_eval", []),
    ("这个房子的 broadband 速度怎么样？", "follow_up", "property_eval", []),
    ("这个房子在 zone 几？", "follow_up", "property_eval", []),
    ("这个房子周围 gentrification 程度如何？", "follow_up", "property_eval", []),
    ("这个房子如果出租，目标租客是谁？", "follow_up", "property_eval", []),
    ("这个房子可以养狗 / 宠物吗？", "follow_up", "property_eval", []),
    ("这个房子的 parking 配套怎么样？", "follow_up", "property_eval", []),
    ("这个房子的服务费过去几年涨了多少？", "follow_up", "property_eval", [{"flat"}]),
    ("这个房子在地震 / 沉降区吗？", "follow_up", "property_eval", []),
    ("这个房子可以做 short-let 吗？", "follow_up", "property_eval", [{"leasehold"}]),

    # ───── follow_up / area_eval (25 — area anchor) ─────
    ("这个区域的人均收入水平如何？", "follow_up", "area_eval", []),
    ("这个区域的失业率怎么样？", "follow_up", "area_eval", []),
    ("这个区域的教育水平如何？", "follow_up", "area_eval", []),
    ("这个区域的犯罪类型主要是什么？", "follow_up", "area_eval", []),
    ("这个区域的人口结构怎么样？", "follow_up", "area_eval", []),
    ("这个区域过去 10 年怎么变化的？", "follow_up", "area_eval", []),
    ("这个区域有大型超市吗？", "follow_up", "area_eval", []),
    ("这个区域 GP / 医疗资源如何？", "follow_up", "area_eval", []),
    ("这个区域的 council 服务质量如何？", "follow_up", "area_eval", []),
    ("这个区域有空气质量问题吗？", "follow_up", "area_eval", []),
    ("这个区域有 noise abatement zone 吗？", "follow_up", "area_eval", []),
    ("这个区域有 conservation area 限制吗？", "follow_up", "area_eval", []),
    ("这个区域的绿地比例如何？", "follow_up", "area_eval", []),
    ("这个区域 ULEZ / congestion 影响大吗？", "follow_up", "area_eval", []),
    ("这个区域 LTN（Low Traffic Neighbourhood）实施得怎么样？", "follow_up", "area_eval", []),
    ("这个区域有 council 重建计划吗？", "follow_up", "area_eval", []),
    ("这个区域 council tax 比附近高吗？", "follow_up", "area_eval", []),
    ("这个区域的 nightlife 多吗？", "follow_up", "area_eval", []),
    ("这个区域适合 staycation 吗？", "follow_up", "area_eval", []),
    ("这个区域 SAH 比例（住自有房比例）高吗？", "follow_up", "area_eval", []),
    ("这个区域 rent vs buy 哪个更划算？", "follow_up", "area_eval", []),
    ("这个区域适合养小孩吗？", "follow_up", "area_eval", []),
    ("这个区域过去一年治安变好还是变差？", "follow_up", "area_eval", []),
    ("这个区域 transit 升级有规划吗？", "follow_up", "area_eval", []),
]


def _matches(anchor_tags: set[str], required_groups: list[set[str]]) -> bool:
    """Anchor matches if its tag set is a superset of at least one required group."""
    if not required_groups:
        return True
    return any(req.issubset(anchor_tags) for req in required_groups)


def _compatible_anchors(intent_subtype: str, required_tags: list[set[str]],
                        anchor_pool: list[str]) -> list[str]:
    if intent_subtype == "property_eval":
        return [a for a in anchor_pool if _matches(PROPERTY_ANCHORS[a]["tags"], required_tags)]
    if intent_subtype == "area_eval":
        return [a for a in anchor_pool if _matches(AREA_ANCHORS[a]["tags"], required_tags)]
    return []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="print stats; don't write output files")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    print(f"New queries to add: {len(NEW_QUERIES)}", file=sys.stderr)
    by_cls = Counter((c, s) for _, c, s, _ in NEW_QUERIES)
    print("By class/subtype:", file=sys.stderr)
    for k, n in by_cls.most_common():
        print(f"  {k}: {n}", file=sys.stderr)
    print(file=sys.stderr)

    # Read existing
    existing_queries = [json.loads(l) for l in
                        (DATA / "queries_filtered.jsonl").read_text().splitlines() if l.strip()]
    existing_review = json.loads((DATA / "queries_review.json").read_text())
    print(f"Existing: {len(existing_queries)} queries / {len(existing_review)} review entries",
          file=sys.stderr)

    # ──────── Step 1: split assignment ────────
    # For each NEW follow_up query, decide whether it goes in train /
    # test_query / test_anchor. Target ratios within follow_up: 60/24/16
    # (adjusted to actual counts):
    #   property_eval: 35 → train 21, test_query 8, test_anchor 6
    #   area_eval:     25 → train 15, test_query 6, test_anchor 4
    # For non-follow_up: 100 → 80 train + 20 test_query (anchor field null).

    followup_indices = [i for i, q in enumerate(NEW_QUERIES) if q[1] == "follow_up"]
    other_indices = [i for i, q in enumerate(NEW_QUERIES) if q[1] != "follow_up"]

    # Stratified split per subtype for follow_up — respects compatibility.
    # If a query is compatible ONLY with hold-out anchors → force test_anchor.
    # If a query CAN'T use any hold-out anchor → exclude from test_anchor.
    def split_followup(indices: list[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        by_sub: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            by_sub[NEW_QUERIES[i][2]].append(i)
        for subtype, idxs in by_sub.items():
            train_pool = TRAIN_PROPERTY_ANCHORS if subtype == "property_eval" else TRAIN_AREA_ANCHORS
            holdout_pool = HOLDOUT_PROPERTY_ANCHORS if subtype == "property_eval" else HOLDOUT_AREA_ANCHORS
            # Classify each query by compatibility
            forced_holdout: list[int] = []   # only holdout anchors fit
            no_holdout: list[int] = []       # holdout anchors don't fit
            flexible: list[int] = []         # both work
            for i in idxs:
                req = NEW_QUERIES[i][3]
                in_train = bool(_compatible_anchors(subtype, req, train_pool))
                in_hold = bool(_compatible_anchors(subtype, req, holdout_pool))
                if in_hold and not in_train:
                    forced_holdout.append(i)
                elif in_train and not in_hold:
                    no_holdout.append(i)
                else:
                    flexible.append(i)
            # Forced go straight into test_anchor
            for i in forced_holdout:
                out[i] = "test_anchor"
            # No-holdout: split between train + test_query only (no test_anchor)
            rng.shuffle(no_holdout)
            n_no = len(no_holdout)
            n_no_test = max(1, round(n_no * 0.24)) if n_no else 0
            for i in no_holdout[:n_no - n_no_test]:
                out[i] = "train"
            for i in no_holdout[n_no - n_no_test:]:
                out[i] = "test_query"
            # Flexible: fill remaining test_anchor quota first, then train/test_query
            n_total = len(idxs)
            target_test_anchor = max(1, round(n_total * 0.16))
            target_test_query = max(1, round(n_total * 0.24))
            still_need_test_anchor = max(0, target_test_anchor - len(forced_holdout))
            already_test_query = sum(1 for v in out.values() if v == "test_query")
            still_need_test_query = max(0, target_test_query - already_test_query)
            rng.shuffle(flexible)
            for i in flexible[:still_need_test_anchor]:
                out[i] = "test_anchor"
            for i in flexible[still_need_test_anchor:still_need_test_anchor + still_need_test_query]:
                out[i] = "test_query"
            for i in flexible[still_need_test_anchor + still_need_test_query:]:
                out[i] = "train"
        return out

    def split_other(indices: list[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        # 80/20 train/test_query for non-followup (no anchor concern)
        # Stratify by class+subtype to be safe
        by_sub: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i in indices:
            by_sub[(NEW_QUERIES[i][1], NEW_QUERIES[i][2])].append(i)
        for key, idxs in by_sub.items():
            shuffled = list(idxs)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_test = max(1, round(n * 0.20))
            n_train = n - n_test
            for i in shuffled[:n_train]:
                out[i] = "train"
            for i in shuffled[n_train:]:
                out[i] = "test_query"
        return out

    splits: dict[int, str] = {}
    splits.update(split_followup(followup_indices))
    splits.update(split_other(other_indices))

    # ──────── Step 2: anchor assignment ────────
    # Round-robin per (subtype, split) bucket within compatible anchors.
    # train + test_query use TRAIN_*_ANCHORS; test_anchor uses HOLDOUT_*_ANCHORS.

    anchor_assignment: dict[int, str | None] = {}
    rr_counters: dict[str, int] = defaultdict(int)  # anchor_id → assigned count

    def pick_anchor(idx: int) -> str | None:
        cls, sub = NEW_QUERIES[idx][1], NEW_QUERIES[idx][2]
        if cls != "follow_up":
            return None
        required = NEW_QUERIES[idx][3]
        split = splits[idx]
        if sub == "property_eval":
            pool = TRAIN_PROPERTY_ANCHORS if split != "test_anchor" else HOLDOUT_PROPERTY_ANCHORS
        elif sub == "area_eval":
            pool = TRAIN_AREA_ANCHORS if split != "test_anchor" else HOLDOUT_AREA_ANCHORS
        else:
            return None
        compat = _compatible_anchors(sub, required, pool)
        if not compat:
            # Fall back: try the full anchor set across BOTH train + holdout
            # so we don't leave a query without an anchor. Print warning.
            full = (TRAIN_PROPERTY_ANCHORS + HOLDOUT_PROPERTY_ANCHORS) if sub == "property_eval" else (TRAIN_AREA_ANCHORS + HOLDOUT_AREA_ANCHORS)
            compat = _compatible_anchors(sub, required, full)
            print(f"  ! query {idx} ({sub}) had no compatible anchor in {pool}; falling back to {compat}", file=sys.stderr)
        # Round-robin: pick the compatible anchor with the lowest current count
        compat_sorted = sorted(compat, key=lambda a: (rr_counters[a], a))
        chosen = compat_sorted[0] if compat_sorted else None
        if chosen:
            rr_counters[chosen] += 1
        return chosen

    for i in range(len(NEW_QUERIES)):
        anchor_assignment[i] = pick_anchor(i)

    # ──────── Step 3: write records ────────
    next_id = 51
    new_queries = []
    new_review = {}
    now = datetime.now(timezone.utc).isoformat()
    for idx, (query_text, cls, subtype, _) in enumerate(NEW_QUERIES):
        cid = f"curated_{next_id:02d}"
        next_id += 1
        anchor_id = anchor_assignment[idx]
        split = splits[idx]
        rec = {
            "candidate_id": cid,
            "primary_query": query_text,
            "title": query_text,
            "source": "curated",
            "source_url": "",
            "raw_body": "",
            "search_term": "curated",
            "search_class_hint": cls,
            "filter_class": cls,
            "filter_confidence": 1.0,
            "intent_subtype": subtype,
            "lang": "zh",
            "mined_at": now,
            "anchor_id": anchor_id,
            "split": split,
        }
        new_queries.append(rec)
        new_review[cid] = {
            "status": "kept",
            "final_class": cls,
            "decided_at": now,
            "split": split,
            "anchor_id": anchor_id,
        }

    # ──────── Step 4: report or write ────────
    print("\n=== Split distribution (NEW only) ===", file=sys.stderr)
    split_counts = Counter(splits.values())
    for s, n in split_counts.most_common():
        print(f"  {s}: {n}", file=sys.stderr)
    print("\n=== Anchor usage (NEW follow_up only) ===", file=sys.stderr)
    for a in TRAIN_PROPERTY_ANCHORS + HOLDOUT_PROPERTY_ANCHORS + TRAIN_AREA_ANCHORS + HOLDOUT_AREA_ANCHORS:
        n = rr_counters.get(a, 0)
        print(f"  {a}: {n}", file=sys.stderr)
    print("\n=== Per-anchor split distribution ===", file=sys.stderr)
    by_anchor_split: dict[str, Counter] = defaultdict(Counter)
    for r in new_queries:
        if r["anchor_id"]:
            by_anchor_split[r["anchor_id"]][r["split"]] += 1
    for a, c in sorted(by_anchor_split.items()):
        line = ", ".join(f"{k}={v}" for k, v in c.most_common())
        print(f"  {a}: {line}", file=sys.stderr)

    if args.dry_run:
        print(f"\nDRY RUN — would add {len(new_queries)} queries (curated_51 → curated_{next_id-1})",
              file=sys.stderr)
        return 0

    out_q = DATA / "queries_filtered.v2.jsonl"
    out_r = DATA / "queries_review.v2.json"
    with out_q.open("w") as f:
        for r in existing_queries:
            # Tag existing 50 — they used the legacy single anchor (the Poplar
            # flat / E14 9NA = A1 / Z1) and were exposed during early baseline runs.
            # Mark all as "train" since the optimization rounds already saw them.
            r["anchor_id"] = (
                "A1" if r.get("intent_subtype") == "property_eval"
                else "Z1" if r.get("intent_subtype") == "area_eval"
                else None
            )
            r["split"] = "train"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in new_queries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    merged_review = {**existing_review, **new_review}
    # Backfill split + anchor on existing review entries too
    for cid, entry in existing_review.items():
        if cid in new_review:
            continue
        entry.setdefault("split", "train")
        # We don't know intent_subtype here without looking at queries — leave anchor null
        merged_review[cid] = entry
    out_r.write_text(json.dumps(merged_review, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_q.name} ({len(existing_queries) + len(new_queries)} queries)",
          file=sys.stderr)
    print(f"Wrote {out_r.name} ({len(merged_review)} review entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
