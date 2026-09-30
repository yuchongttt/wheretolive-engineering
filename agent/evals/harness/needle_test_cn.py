#!/usr/bin/env python3
"""Chinese-language run of the needle test — same targets, queries written in
Chinese, lang='zh' so the agent uses the Chinese system prompt and replies in
Chinese. Tests whether retrieval quality holds when the user writes in Chinese.
The Chinese query strings are the test input and are kept verbatim; each is a
translation of the matching English needle in needle_test.py. Target ids and
streets are redacted placeholders in the public copy (see needle_test.py).

Hub names are kept in English (Stratford / Canary Wharf / Clapham Junction) —
that's how London-based Chinese speakers usually write them, and the commute
tools key on the English hub strings. Everything else is natural Chinese.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle_test import run_case, _admin_key, replay  # noqa: E402

NEEDLES = [
    {
        "name": "stratford-garden-house",
        "target_id": "<listing-id-1>",
        "street": "<street>",
        "query": ("我想找一套 4 卧的独立屋（detached），永久产权，要有私人花园和至少 3 "
                  "个客厅（reception rooms），外加 3 个卫生间，通勤到 Stratford 大约 40 "
                  "分钟以内。预算 125 万英镑左右（110 万到 140 万）。没有特定的区域偏好。"),
    },
    {
        "name": "canarywharf-decor-flat",
        "target_id": "<listing-id-2>",
        "street": "<street>",
        "query": ("帮我找一套 2 卧 2 卫的公寓（apartment），室内装修要顶级——最好是市面上"
                  "装修最好的那种，最好带阳台，距离 Canary Wharf 15 分钟以内。预算 87.5 "
                  "万英镑左右，最高 95 万。"),
    },
    {
        "name": "claphamjct-garden-decor-house",
        "target_id": "<listing-id-3>",
        "street": "<street>",
        "query": ("我想要一套大的 5 卧房子（house），永久产权，带花园，至少 3 个客厅，室内装修精美"
                  "（顶级装修档次），距离 Clapham Junction 20 分钟以内。预算 240 万英镑左右"
                  "（220 万到 260 万）。不限定具体房型。"),
    },
    {
        "name": "garage-house-claphamjct",
        "target_id": "<listing-id-4>",
        "street": "<street>",
        "query": ("想找一套 3 卧的房子（house，永久产权），带车库，通勤到 Clapham Junction "
                  "30 分钟以内，预算 50 万英镑左右（45 万到 55 万）。不限定区域。"),
    },
    {
        "name": "spacious-3bed-canarywharf",
        "target_id": "<listing-id-5>",
        "street": "<street>",
        "query": ("我想要一套特别宽敞的 3 卧——室内面积至少 1800 平方英尺——距离 Canary "
                  "Wharf 20 分钟以内，预算 75 万英镑左右，最高 80 万。"),
    },
    {
        "name": "decor-pick-liverpoolst",
        "target_id": "<listing-id-6>",
        "street": "<street>",
        "query": ("帮我找一套距离 Liverpool Street 15 分钟以内、装修最好的 2 卧公寓，预算"
                  "最高 90 万——我最看重室内装修和质感。"),
    },
    {
        "name": "garden-receptions-ealing",
        "target_id": "<listing-id-7>",
        "street": "<street>",
        "query": ("我想要一套 4 卧的房子（house，永久产权），带花园，至少 3 个客厅，距离 "
                  "Ealing Broadway 25 分钟以内，预算 130 万英镑左右（120 万到 140 万）。"),
    },
]


def main():
    admin_key = _admin_key()
    out = []
    for nd in NEEDLES:
        print(f"\n=== {nd['name']} (target {nd['target_id']}) [zh] ===")
        print(f"  Q: {nd['query']}")
        res = run_case({"query": nd["query"], "lang": "zh"}, admin_key, timeout=300)

        recall_hit, replay_detail = False, []
        for tc in res["tool_calls"]:
            if tc["name"] not in ("search_properties", "search_floorplans"):
                continue
            hit, n = replay(tc["name"], tc["args"], nd["target_id"])
            if hit is None:
                continue
            replay_detail.append({"tool": tc["name"], "result_count": n, "target_in_results": hit})
            recall_hit = recall_hit or hit

        ans = res["answer"]
        exact = nd["target_id"] in ans
        street_soft = nd["street"].lower() in ans.lower()
        # crude check that the reply is actually in Chinese
        cjk = sum(1 for ch in ans if '一' <= ch <= '鿿')

        for tc in res["tool_calls"]:
            arg_s = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
            print(f"  -> {tc['name']}({arg_s})")
        for rd in replay_detail:
            print(f"     replay {rd['tool']}: {rd['result_count']} results, target_in={rd['target_in_results']}")
        print(f"  RECALL: {recall_hit}   EXACT SURFACED: {exact}   (street-soft={street_soft})")
        print(f"  reply CJK chars: {cjk} ({'zh' if cjk > 200 else 'NOT zh?'}), {len(ans)} chars, {res['elapsed_s']}s, errors={res['errors']}")

        out.append({"name": nd["name"], "target_id": nd["target_id"], "query": nd["query"],
                    "tool_calls": res["tool_calls"], "replay": replay_detail,
                    "recall_hit": recall_hit, "exact_surfaced": exact, "street_soft": street_soft,
                    "cjk_chars": cjk, "answer": ans, "elapsed_s": res["elapsed_s"], "errors": res["errors"]})

    nrecall = sum(1 for o in out if o["recall_hit"])
    nexact = sum(1 for o in out if o["exact_surfaced"])
    print(f"\n{'='*50}")
    print(f"RECALL:        {nrecall}/{len(out)}")
    print(f"EXACT SURFACED: {nexact}/{len(out)}")
    Path("/tmp/needle_cn_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("results -> /tmp/needle_cn_results.json")


if __name__ == "__main__":
    main()
