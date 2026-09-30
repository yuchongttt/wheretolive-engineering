#!/usr/bin/env python3
"""
Price analysis evaluator
Analyses local house-price trends using the Land Registry and EPC APIs
"""

import re
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

from api_helper import LandRegistryAPI, LocalEPCService, load_api_params


class PriceAnalysisEvaluator:
    """Price analysis evaluator"""

    # Property type labels
    PROPERTY_TYPE_NAMES = {
        "D": "Detached",
        "S": "Semi-detached",
        "T": "Terraced",
        "F": "Flat",
        "O": "Other"
    }

    # Tenure type labels
    TENURE_TYPE_NAMES = {
        "F": "Freehold",
        "L": "Leasehold",
        "U": "Unknown"
    }

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialise the price analysis evaluator

        Args:
            params: API parameter config
        """
        if params is None:
            params = load_api_params()

        self.params = params
        self.land_registry_api = LandRegistryAPI()

        # The local EPC microservice (the GPU box :8400; full dataset, refreshed monthly from
        # the new GOV.UK service) is the only source of £/m². The old official-API fallback
        # (epc.opendatacommunities.org) went offline on 2026-05-30 and also retried with waits
        # on failure, so it was removed. If the service is down, search returns None and the
        # evaluation carries on as normal (just without £/m²).
        epc_config = params.get("epc", {})
        self.epc_api = None
        self.epc_local = LocalEPCService(epc_config.get("service_url"))

    def _extract_postcode(self, address: str) -> Optional[str]:
        """
        Extract the postcode from an address

        Args:
            address: full address string

        Returns:
            Postcode string, or None if not found
        """
        # UK postcode regex
        # Formats: A9 9AA, A99 9AA, AA9 9AA, AA99 9AA, A9A 9AA, AA9A 9AA
        postcode_pattern = r'\b([A-Z]{1,2}[0-9][0-9A-Z]?\s*[0-9][A-Z]{2})\b'

        match = re.search(postcode_pattern, address.upper())
        if match:
            postcode = match.group(1)
            # Normalise the format (ensure a space in the middle)
            postcode = postcode.replace(" ", "")
            if len(postcode) > 3:
                return f"{postcode[:-3]} {postcode[-3:]}"
            return postcode

        return None

    def _identify_repeat_sales(
        self,
        transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify repeat sales (multiple transactions at the same address)

        Args:
            transactions: list of transaction records

        Returns:
            List of repeat-sale records
        """
        # Group by address
        by_address = defaultdict(list)
        for t in transactions:
            addr_key = t["address"].lower().strip()
            by_address[addr_key].append(t)

        repeat_sales = []
        for addr, sales in by_address.items():
            if len(sales) >= 2:
                # Sort by date
                sorted_sales = sorted(sales, key=lambda x: x["date"])

                for i in range(1, len(sorted_sales)):
                    prev = sorted_sales[i - 1]
                    curr = sorted_sales[i]

                    # LR category (optionally supplied by the caller): pairing must run over the
                    # full sale sequence, and pairs with a B leg are dropped afterwards. Dropping
                    # rows first and then pairing would stitch A→B→A chains (10,673 in the full
                    # DB) into phantom holdings spanning different owners (R3-1-refix).
                    # Callers without a category key behave as before.
                    if prev.get("category") == "B" or curr.get("category") == "B":
                        continue

                    # Compute the price change
                    if prev["price"] > 0:
                        price_change = curr["price"] - prev["price"]
                        price_change_pct = (price_change / prev["price"]) * 100

                        # Compute the holding period (years)
                        try:
                            prev_date = datetime.strptime(prev["date"][:10], "%Y-%m-%d")
                            curr_date = datetime.strptime(curr["date"][:10], "%Y-%m-%d")
                            years_held = (curr_date - prev_date).days / 365.25
                        except (ValueError, TypeError):
                            years_held = 0

                        # Skip pairs held < 3 months (likely administrative transfers, name corrections, etc.)
                        if years_held < 0.25:
                            continue

                        # Compute the annualised growth rate
                        cagr = ((curr["price"] / prev["price"]) ** (1 / years_held) - 1) * 100
                        # Clamp the annualised return to ±100% so short holds do not produce extreme values
                        cagr = max(-100, min(100, cagr))

                        repeat_sales.append({
                            "address": curr["address"],
                            "previous_sale": {
                                "date": prev["date"],
                                "price": prev["price"]
                            },
                            "current_sale": {
                                "date": curr["date"],
                                "price": curr["price"]
                            },
                            "price_change": price_change,
                            "price_change_pct": round(price_change_pct, 2),
                            "years_held": round(years_held, 2),
                            "annualized_return": round(cagr, 2)
                        })

        return repeat_sales

    @staticmethod
    def _select_repeat_sales(repeat_sales: List[Dict[str, Any]], total: int = 5) -> List[Dict[str, Any]]:
        """
        Pick representative repeat sales: up to 3 most recent, remaining slots go to the longest held.
        Each entry carries selection_reason ("recent"/"long_held") so the frontend can label how it was picked;
        each address appears at most once (matching what the frontend tooltip promises);
        the address key is lower().strip(), same as in _identify_repeat_sales.
        """
        if not repeat_sales:
            return []

        used_addrs = set()

        def take(candidates: List[Dict[str, Any]], reason: str, quota: int) -> List[Dict[str, Any]]:
            picked = []
            for rs in candidates:
                addr = rs["address"].lower().strip()
                if addr in used_addrs:
                    continue
                picked.append({**rs, "selection_reason": reason})
                used_addrs.add(addr)
                if len(picked) >= quota:
                    break
            return picked

        by_recent = sorted(repeat_sales, key=lambda x: x["current_sale"]["date"], reverse=True)
        recent = take(by_recent, "recent", min(3, total))

        by_span = sorted(repeat_sales, key=lambda x: x["years_held"], reverse=True)
        long_span = take(by_span, "long_held", total - len(recent))

        return recent + long_span

    def _calculate_price_trends(
        self,
        transactions: List[Dict[str, Any]],
        years: int = 10,
        repeat_sales: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute price trends (based on repeat-sale data)

        Args:
            transactions: list of transaction records
            years: number of years to analyse
            repeat_sales: list of repeat-sale records

        Returns:
            Trend analysis result
        """
        current_year = datetime.now().year

        # No repeat-sale data: return an empty result
        if not repeat_sales:
            return {
                "10_year_cagr": None,
                "5_year_cagr": None,
                "3_year_cagr": None,
                "direction": "unknown",
                "sample_count": 0
            }

        def calculate_avg_cagr(years_back: int) -> tuple:
            """Average annualised return over the given number of years (weighted by holding period)"""
            cutoff_year = current_year - years_back
            # Keep repeat sales completed within the window
            filtered = [
                rs for rs in repeat_sales
                if int(rs["current_sale"]["date"][:4]) >= cutoff_year
            ]
            if not filtered:
                return None, 0
            # Weight by holding period: longer holds get more weight
            total_weighted = sum(rs["annualized_return"] * rs["years_held"] for rs in filtered)
            total_years = sum(rs["years_held"] for rs in filtered)
            if total_years <= 0:
                return None, 0
            weighted_avg = total_weighted / total_years
            return round(weighted_avg, 2), len(filtered)

        # Average annualised return for each period
        cagr_10, count_10 = calculate_avg_cagr(10)
        cagr_5, count_5 = calculate_avg_cagr(5)
        cagr_3, count_3 = calculate_avg_cagr(3)

        # Determine the trend direction (from recent repeat sales, weighted by holding period)
        recent_sales = [
            rs for rs in repeat_sales
            if int(rs["current_sale"]["date"][:4]) >= current_year - 3
        ]
        if recent_sales:
            total_weighted = sum(rs["annualized_return"] * rs["years_held"] for rs in recent_sales)
            total_years = sum(rs["years_held"] for rs in recent_sales)
            avg_recent = total_weighted / total_years if total_years > 0 else 0
            if avg_recent > 3:
                direction = "up"
            elif avg_recent < -3:
                direction = "down"
            else:
                direction = "stable"
        else:
            direction = "unknown"

        return {
            "10_year_cagr": cagr_10,
            "5_year_cagr": cagr_5,
            "3_year_cagr": cagr_3,
            "10_year_count": count_10,
            "5_year_count": count_5,
            "3_year_count": count_3,
            "direction": direction,
            "sample_count": len(repeat_sales)
        }

    def _extract_address_features(self, address: str) -> Dict[str, Optional[str]]:
        """
        Extract several address features for matching

        Args:
            address: address string

        Returns:
            Dict with unit_number, building_name, street_number, street_name
        """
        addr = address.upper().strip()
        features = {
            "unit_number": None,      # FLAT/UNIT/APARTMENT number
            "building_name": None,    # building name
            "street_number": None,    # street (house) number
            "street_name": None,      # street name
        }

        # 1. Extract the FLAT/UNIT/APARTMENT number (keyword-prefixed)
        unit_match = re.search(r'(?:FLAT|UNIT|APARTMENT|APT)\s*(\d+[A-Z]?)', addr)
        if unit_match:
            features["unit_number"] = unit_match.group(1)

        # 2. Extract the building name (common patterns: XXX TOWER, XXX APARTMENTS, XXX HOUSE, XXX COURT)
        building_patterns = [
            r'([A-Z][A-Z\s]+(?:TOWER|APARTMENTS|HOUSE|COURT|MANSIONS|LODGE|HALL|BUILDING|POINT|HEIGHTS))',
            r'([A-Z][A-Z\s]+(?:SQUARE|GARDENS|TERRACE|MEWS|RISE|VIEW|WALK)S?)\b',
        ]
        for pattern in building_patterns:
            building_match = re.search(pattern, addr)
            if building_match:
                # Clean up the building name (collapse extra whitespace)
                building = ' '.join(building_match.group(1).split())
                features["building_name"] = building
                break

        # 3. No FLAT/UNIT keyword, but the address starts with a number followed by the building name
        #    e.g. "12, EXAMPLE COURT" -> unit_number = 12
        if not features["unit_number"] and features["building_name"]:
            # Check for the "number, building name" or "number building name" format
            first_num_match = re.match(r'^(\d+[A-Z]?)\s*[,\s]\s*' + re.escape(features["building_name"]), addr)
            if first_num_match:
                features["unit_number"] = first_num_match.group(1)
            else:
                # Or the number appears somewhere before the building name
                parts = addr.split(',')
                for i, part in enumerate(parts):
                    part = part.strip()
                    if re.match(r'^\d+[A-Z]?$', part):
                        # Check whether the next part contains the building name
                        if i + 1 < len(parts) and features["building_name"] in parts[i + 1]:
                            features["unit_number"] = part
                            break

        # 4. Extract the street number and street name
        parts = [p.strip() for p in addr.split(',')]

        for i, part in enumerate(parts):
            # Skip the already-identified FLAT/UNIT part
            if re.match(r'^(?:FLAT|UNIT|APARTMENT|APT)\s*\d+', part):
                continue
            # Skip the part already identified as unit_number
            if features["unit_number"] and re.match(r'^' + re.escape(features["unit_number"]) + r'$', part):
                continue
            # Skip the already-identified building name
            if features["building_name"] and features["building_name"] in part:
                continue

            # Try to match "number street-name"
            street_match = re.match(r'^(\d+[A-Z]?)\s+([A-Z][A-Z\s]+(?:ROAD|STREET|LANE|AVENUE|WAY|DRIVE|CLOSE|CRESCENT|GROVE|PLACE|HILL))\b', part)
            if street_match:
                features["street_number"] = street_match.group(1)
                features["street_name"] = street_match.group(2).strip()
                break

            # Standalone street name
            street_name_match = re.search(r'\b([A-Z][A-Z\s]+(?:ROAD|STREET|LANE|AVENUE|WAY|DRIVE|CLOSE|CRESCENT|GROVE|PLACE|HILL))\b', part)
            if street_name_match and not features["street_name"]:
                features["street_name"] = street_name_match.group(1).strip()

            # A standalone number (after the building name) may be the street number
            if re.match(r'^\d+[A-Z]?$', part) and not features["street_number"]:
                # The number is a street number only if the building name has already appeared
                prev_parts = ','.join(parts[:i]).upper()
                if features["building_name"] and features["building_name"] in prev_parts:
                    features["street_number"] = part

        # 5. "50, EXAMPLE ROAD" form: the house number is its own segment, which the same-segment
        #    regex above misses. If a street name was found but no street/unit number, check
        #    whether the segment before the street name is purely numeric
        if (features["street_name"] and not features["street_number"]
                and not features["unit_number"]):
            for i, part in enumerate(parts):
                if features["street_name"] in part:
                    if i > 0 and re.match(r'^\d+[A-Z]?$', parts[i - 1]):
                        features["street_number"] = parts[i - 1]
                    break

        return features

    def _normalize_address_key(self, address: str) -> Optional[str]:
        """
        Extract a matching key from the address (backward compatible)

        Args:
            address: address string

        Returns:
            Normalised address key used for matching
        """
        features = self._extract_address_features(address)

        # Prefer unit_number
        if features["unit_number"]:
            return f"FLAT{features['unit_number']}"

        # Then street_number
        if features["street_number"]:
            return features["street_number"]

        # Finally, try to extract a number from the start of the address
        addr = address.upper().strip()
        first_part = addr.split(',')[0].strip()
        number_in_first = re.search(r'\b(\d+[A-Z]?)\b', first_part)
        if number_in_first:
            return number_in_first.group(1)

        return None

    def _calculate_address_match_score(
        self,
        features1: Dict[str, Optional[str]],
        features2: Dict[str, Optional[str]]
    ) -> int:
        """
        Compute a match score between two sets of address features

        Args:
            features1: features of the first address
            features2: features of the second address

        Returns:
            Match score (0-4); higher is better
        """
        score = 0

        # unit_number must match (if both have one)
        if features1["unit_number"] and features2["unit_number"]:
            if features1["unit_number"] == features2["unit_number"]:
                score += 2  # unit-number match carries a high weight
            else:
                return 0  # different unit numbers: no match

        # building_name match
        if features1["building_name"] and features2["building_name"]:
            # Compare after removing generic words
            common_suffixes = {"TOWER", "APARTMENTS", "HOUSE", "COURT", "MANSIONS",
                              "LODGE", "HALL", "BUILDING", "POINT", "HEIGHTS",
                              "SQUARE", "GARDENS", "TERRACE", "MEWS", "THE", "OF", "AT"}
            words1 = set(features1["building_name"].split()) - common_suffixes
            words2 = set(features2["building_name"].split()) - common_suffixes
            # Require a substantive name match (not just the suffix)
            if words1 and words2:
                common_words = words1 & words2
                if common_words:
                    score += 1
                else:
                    # Both have building names but the main names differ: cannot pair
                    return 0
            elif not words1 and not words2:
                # Both are suffix-only (e.g. "THE APARTMENTS"): allow
                pass
            else:
                # One has a main name and the other does not: no match
                return 0

        # street_number match
        if features1["street_number"] and features2["street_number"]:
            if features1["street_number"] == features2["street_number"]:
                score += 1

        # street_name match
        if features1["street_name"] and features2["street_name"]:
            # Simple containment check
            if (features1["street_name"] in features2["street_name"] or
                features2["street_name"] in features1["street_name"]):
                score += 1

        return score

    def _merge_with_epc_data(
        self,
        transactions: List[Dict[str, Any]],
        epc_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge transaction data with EPC data (multi-feature matching)

        Args:
            transactions: list of transaction records
            epc_records: list of EPC records

        Returns:
            List of merged transaction records
        """
        if not epc_records:
            return transactions

        # Preprocess EPC records: extract features
        epc_with_features = []
        for epc in epc_records:
            addr = epc.get("address", "")
            features = self._extract_address_features(addr)
            epc_with_features.append({
                "epc": epc,
                "features": features,
                "key": self._normalize_address_key(addr)
            })

        # Build lookup indexes (by unit_number, key, postcode)
        epc_by_unit = {}
        epc_by_key = {}
        epc_by_postcode = {}
        for item in epc_with_features:
            unit = item["features"]["unit_number"]
            key = item["key"]
            if unit:
                if unit not in epc_by_unit:
                    epc_by_unit[unit] = []
                epc_by_unit[unit].append(item)
            if key:
                if key not in epc_by_key:
                    epc_by_key[key] = []
                epc_by_key[key].append(item)
            pc = (item["epc"].get("postcode") or "").replace(" ", "").upper()
            if pc:
                epc_by_postcode.setdefault(pc, []).append(item)

        # Merge data
        merged = []
        matched_count = 0
        high_confidence_count = 0

        for t in transactions:
            merged_t = t.copy()
            addr = t.get("address", "")
            t_features = self._extract_address_features(addr)
            t_key = self._normalize_address_key(addr)

            # Find candidate EPC records
            candidates = []
            if t_features["unit_number"] and t_features["unit_number"] in epc_by_unit:
                candidates.extend(epc_by_unit[t_features["unit_number"]])
            elif t_key and t_key in epc_by_key:
                candidates.extend(epc_by_key[t_key])
            if not candidates:
                # No key match (LR addresses often carry a town-name suffix) → score every EPC
                # record in the same postcode. ≤~50 per postcode, so the cost is bounded; the score
                # threshold (≥2) is unchanged, so precision does not drop and only recall improves.
                t_pc = (t.get("postcode") or "").replace(" ", "").upper()
                if t_pc and t_pc in epc_by_postcode:
                    candidates = epc_by_postcode[t_pc]

            # Score the candidates and pick the best match
            best_match = None
            best_score = 0
            for item in candidates:
                score = self._calculate_address_match_score(t_features, item["features"])
                if score > best_score:
                    best_score = score
                    best_match = item["epc"]

            # Only accept matches with score >= 2 (at least the unit number, or several other features)
            if best_match and best_score >= 2:
                floor_area = best_match.get("floor_area", 0)
                if floor_area and floor_area > 0:
                    merged_t["floor_area"] = floor_area
                    merged_t["epc_rating"] = best_match.get("current_energy_rating", "")
                    merged_t["match_score"] = best_score
                    if t["price"] > 0:
                        merged_t["price_per_sqm"] = round(t["price"] / floor_area, 2)
                    matched_count += 1
                    if best_score >= 3:
                        high_confidence_count += 1

            merged.append(merged_t)

        if matched_count > 0:
            print(f"  ✅ Matched floor-area data for {matched_count} transactions (high confidence: {high_confidence_count})")

        return merged

    # Residential types (commercial property excluded)
    RESIDENTIAL_TYPES = {"D", "S", "T", "F"}  # Detached, Semi-detached, Terraced, Flat
    MIN_VALID_TXNS = 8  # Fewer valid residential sales than this in the unit postcode → fall back to sector/outcode in the local DB
    EPC_MAX_POSTCODES = 40  # Max number of postcodes to fetch EPC for when the area is widened (top K by sales volume)
    MIN_SQM_SAMPLE = 5  # Fewer £/m² matches than this → no figure (same rule as the frontend's MIN_SAMPLE)
    # Nominal-transfer gate for repeat-sale pairing: £1/£100-level LR records are family
    # transfers / title adjustments, not market sales; pairing them with a real resale
    # produces fake +100%/yr pairs.
    # Threshold of £10k: below the real floor price of London homes since 1995, but high
    # enough to block nominal transfers.
    PAIRING_MIN_PRICE = 10_000
                        # (so sparse/commercial unit postcodes such as SW7 1AY don't fail the whole price dimension)

    def _filter_residential_only(
        self,
        transactions: List[Dict[str, Any]]
    ) -> tuple:
        """
        Keep only residential transactions (exclude commercial property)

        Args:
            transactions: list of transaction records

        Returns:
            (filtered transactions, excluded transactions)
        """
        filtered = []
        excluded = []
        for t in transactions:
            if t.get("property_type", "O") in self.RESIDENTIAL_TYPES:
                filtered.append(t)
            else:
                excluded.append(t)
        return filtered, excluded

    def _filter_market_sales_only(
        self,
        transactions: List[Dict[str, Any]]
    ) -> tuple:
        """
        Keep only standard market sales (LR category A).

        Category B = "additional price paid": company purchases / repossessions / power-of-sale
        and other non-standard transfers, about 18% of recent sales; overall they pull the
        outcode median down by ~3% (±30% at the extremes). Statistics (median / CAGR /
        repeat-sale pairing) use category A only; a missing category (old data, not returned
        by SPARQL) counts as A, matching the IS NULL handling in the web app's lrMarketStatsWhere.

        Returns:
            (kept transactions, excluded category-B records)
        """
        filtered = []
        excluded = []
        for t in transactions:
            if t.get("category") == "B":
                excluded.append(t)
            else:
                filtered.append(t)
        return filtered, excluded

    def _local_db_path(self) -> str:
        return str(Path(__file__).parent / "data" / "evaluations.db")

    def _residential_count(self, transactions: List[Dict[str, Any]]) -> int:
        """Count valid residential sales (same rule as _filter_residential_only +
        _filter_market_sales_only: residential type and not category B), so the sample size
        the fallback decision sees equals the sample actually usable after filtering."""
        return sum(1 for t in transactions
                   if t.get("property_type", "O") in self.RESIDENTIAL_TYPES
                   and t.get("category") != "B")

    def _fetch_local_area_transactions(
        self, outcode: str, sector_char: Optional[str], years: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch sales from the local lr_transactions table by sector (outcode + sector digit)
        or by outcode, returning exactly the same dict shape as the SPARQL path so the
        existing residential/outlier filters can be reused.
        """
        cutoff = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        np_sql = "replace(upper(postcode), ' ', '')"
        where = (
            f"{np_sql} != '' AND length({np_sql}) >= 5 AND price > 0 AND date >= ? "
            f"AND substr({np_sql}, 1, length({np_sql}) - 3) = ?"
        )
        params: List[Any] = [cutoff, outcode]
        if sector_char is not None:
            where += f" AND substr({np_sql}, length({np_sql}) - 2, 1) = ?"
            params.append(sector_char)
        try:
            conn = sqlite3.connect(
                f"file:{self._local_db_path()}?mode=ro", uri=True, timeout=30
            )
        except sqlite3.Error as e:
            print(f"   ⚠️  Failed to connect to the local sales DB: {e}")
            return []
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT postcode, paon, saon, street, price, date, property_type, "
                f"tenure, new_build, category FROM lr_transactions WHERE {where}",
                params,
            ).fetchall()
        except sqlite3.Error as e:
            print(f"   ⚠️  Local sales query failed: {e}")
            return []
        finally:
            conn.close()
        out = []
        for r in rows:
            parts = [p for p in (r["saon"], r["paon"], r["street"]) if p]
            out.append({
                "address": ", ".join(parts) if parts else "Unknown",
                "postcode": r["postcode"],
                "price": int(r["price"]),
                "date": r["date"] or "",
                "property_type": r["property_type"] or "O",
                "tenure": r["tenure"] or "U",
                "new_build": (r["new_build"] == "Y"),
                "category": r["category"],
            })
        return out

    def _fetch_transactions_with_fallback(self, postcode: str, years: int) -> tuple:
        """
        Primary path: SPARQL on the exact unit postcode (unchanged for postcodes with enough data).
        Fallback: when valid residential sales in the unit postcode < MIN_VALID_TXNS, widen via the
        local lr_transactions to sector → outcode until the sample is large enough.
        Returns (transactions, geo_scope),
        geo_scope = {"scope": "unit"|"sector"|"outcode", "postcode": <actual scope>}.
        """
        unit_txns = self.land_registry_api.get_transactions_by_postcode(
            postcode, years=years
        ) or []
        if self._residential_count(unit_txns) >= self.MIN_VALID_TXNS:
            return unit_txns, {"scope": "unit", "postcode": postcode}

        np = postcode.upper().replace(" ", "")
        if len(np) < 5:
            return unit_txns, {"scope": "unit", "postcode": postcode}
        outcode = np[:-3]
        sector_char = np[-3]
        ladder = [
            ("sector", f"{outcode} {sector_char}", outcode, sector_char),
            ("outcode", outcode, outcode, None),
        ]
        # Keep the unit result by default; widen step by step, keeping the widest non-empty result as a fallback
        widest = (unit_txns, {"scope": "unit", "postcode": postcode})
        for scope_name, scope_pc, oc, sc in ladder:
            area_txns = self._fetch_local_area_transactions(oc, sc, years)
            if area_txns:
                widest = (area_txns, {"scope": scope_name, "postcode": scope_pc})
            if self._residential_count(area_txns) >= self.MIN_VALID_TXNS:
                return area_txns, {"scope": scope_name, "postcode": scope_pc}
        return widest

    def _filter_outliers(
        self,
        transactions: List[Dict[str, Any]]
    ) -> tuple:
        """
        Filter obvious outliers (sales priced outside 17%~600% of the overall median)

        Args:
            transactions: list of transaction records

        Returns:
            (filtered transactions, excluded transactions, median, valid range)
        """
        if len(transactions) < 3:
            return transactions, [], 0, (0, 0)

        prices = [t["price"] for t in transactions if t["price"] > 0]
        if not prices:
            return transactions, [], 0, (0, 0)

        sorted_prices = sorted(prices)
        median_price = sorted_prices[len(sorted_prices) // 2]

        # Compute the valid price range
        min_valid = median_price * 0.17  # 17% of the median
        max_valid = median_price * 6.0  # 600% of the median

        filtered = []
        excluded = []
        for t in transactions:
            if min_valid <= t["price"] <= max_valid:
                filtered.append(t)
            else:
                excluded.append(t)

        return filtered, excluded, median_price, (min_valid, max_valid)

    def analyze_area_prices(
        self,
        postcode: str,
        years: int = 30,
        target_price: int = None,
        filter_outliers: bool = True,
        display_years: int = 10,
        avg_price_years: int = None
    ) -> Dict[str, Any]:
        """
        Analyse local house prices

        Args:
            postcode: postcode
            years: years of data to fetch (default 30)
            target_price: target price (for comparison)
            filter_outliers: whether to filter outliers (recommended off for long-term analysis)
            display_years: number of years shown in the yearly stats (default 10)
            avg_price_years: number of years used for the average price (default None = all data)

        Returns:
            Analysis result dict
        """
        print(f"\n{'='*70}")
        print(f"Price analysis: {postcode}")
        print(f"{'='*70}")
        print(f"Analysis window: last {years} years")
        if target_price:
            print(f"Target price: £{target_price:,}")
        print(f"{'='*70}\n")

        # 1. Fetch Land Registry transactions (exact unit postcode first; if there are too few
        #    residential sales, fall back to sector/outcode in the local lr_transactions so that
        #    sparse/commercial unit postcodes don't fail)
        print("📊 Fetching Land Registry transactions...")
        transactions, geo_scope = self._fetch_transactions_with_fallback(postcode, years)

        if not transactions:
            print("  ❌ No transactions found")
            return {
                "postcode": postcode,
                "error": "No transactions found",
                "summary": None
            }

        original_count = len(transactions)
        if geo_scope.get("scope") != "unit":
            print(f"  ↔ Too few sales in the unit postcode; fell back to {geo_scope['scope']} area "
                  f"({geo_scope['postcode']})")
        print(f"  ✅ Found {original_count} transactions")

        # 2. Filter: keep residential only, drop outliers
        print("\n🔍 Filtering data...")

        # 2.1 Exclude commercial property
        transactions, excluded_commercial = self._filter_residential_only(transactions)
        # 2.1b Exclude non-standard sales (LR category B: company purchases / repossessions /
        # power-of-sale). Statistics and pairing use category A market sales only: category B
        # pulls the median down by ~3% overall, and pairing repossessions / discounted transfers
        # with market sales produces fake gain/loss pairs.
        transactions, excluded_nonstandard = self._filter_market_sales_only(transactions)
        if excluded_nonstandard:
            print(f"  📋 Excluded non-standard sales (category B): {len(excluded_nonstandard)} txns")
        # Repeat-sale pairing uses residential sales from before outlier filtering: the outlier
        # band from a 30-year mixed median would wrongly drop genuine low-priced early sales and
        # break up long-held pairs. Nominal transfers (£1/£100 transfers) must still be blocked:
        # paired with market sales they produce fake +100%/yr pairs that pollute
        # stability/trend/notable_gains
        pairing_transactions = [
            t for t in transactions if t["price"] >= self.PAIRING_MIN_PRICE
        ]
        if excluded_commercial:
            print(f"\n  📋 Excluded non-residential: {len(excluded_commercial)} txns")
            # Show only the first 3 examples
            for t in excluded_commercial[:3]:
                ptype = t.get("property_type", "?")
                ptype_name = self.PROPERTY_TYPE_NAMES.get(ptype, ptype)
                print(f"     e.g. £{t['price']:,} | {ptype_name} | {t['address'][:40]}")
            if len(excluded_commercial) > 3:
                print(f"     ... {len(excluded_commercial)} in total")

        # 2.2 Exclude outliers (optional)
        if filter_outliers:
            transactions, excluded_outliers, median, valid_range = self._filter_outliers(transactions)
            if excluded_outliers:
                # Break down by direction
                too_low = [t for t in excluded_outliers if t['price'] < valid_range[0]]
                too_high = [t for t in excluded_outliers if t['price'] >= valid_range[1]]

                print(f"\n  📋 Excluded outliers: {len(excluded_outliers)} txns (valid range: £{valid_range[0]:,.0f} ~ £{valid_range[1]:,.0f})")

                if too_high:
                    print(f"     Too high ({len(too_high)} txns):")
                    for t in sorted(too_high, key=lambda x: x['price'], reverse=True)[:2]:
                        pct = (t['price'] / median - 1) * 100
                        print(f"       e.g. £{t['price']:,} ({pct:+.0f}%) | {t['address'][:35]}")

                if too_low:
                    print(f"     Too low ({len(too_low)} txns):")
                    for t in sorted(too_low, key=lambda x: x['price'])[:2]:
                        pct = (t['price'] / median - 1) * 100
                        print(f"       e.g. £{t['price']:,} ({pct:+.0f}%) | {t['address'][:35]}")
        else:
            print(f"\n  ⚠️  Outlier filtering disabled (long-term analysis mode)")

        print(f"\n  ✅ Valid transactions: {len(transactions)} in total")

        if not transactions:
            print("  ❌ No valid transactions after filtering")
            return {
                "postcode": postcode,
                "error": "No valid transactions after filtering",
                "summary": None
            }

        # 3. Fetch EPC data (local microservice)
        # When widened to sector/outcode, sales span many postcodes: fetch EPC for the top K
        # postcodes by sales volume (K is bounded so the service isn't overwhelmed). £/m² is
        # computed only from the subset with a successful address-level match; its sample size is
        # sent separately as price_per_sqm_count and is never passed off as covering all sales.
        epc_records = None
        print("\n📋 Fetching EPC data...")
        if geo_scope.get("scope") == "unit":
            epc_postcodes = [postcode]
        else:
            pc_counts = defaultdict(int)
            for t in transactions:
                pc = (t.get("postcode") or "").strip()
                if pc:
                    pc_counts[pc] += 1
            epc_postcodes = [
                pc for pc, _ in sorted(
                    pc_counts.items(), key=lambda kv: kv[1], reverse=True
                )[:self.EPC_MAX_POSTCODES]
            ]
        if epc_postcodes:
            epc_records = self.epc_local.search_by_postcodes(epc_postcodes)
            if epc_records:
                print(f"  ✅ Local EPC service: {len(epc_records)} records "
                      f"({len(epc_postcodes)} postcodes)")

        # 4. Merge data
        if epc_records:
            transactions = self._merge_with_epc_data(transactions, epc_records)

        # 5. Compute statistics
        print("\n📈 Computing statistics...")

        # Transactions used for the average price: widen the window dynamically so the sample
        # is >= MIN_AVG_SAMPLE (unit postcodes are fine-grained, a fixed 5-year window often has
        # only 1-2 sales, and treating a single sale as the "area average" is badly misleading.
        # Now: 5y first → if not enough, 10y → 20y → all 30y.)
        MIN_AVG_SAMPLE = 5
        EXPANSION_LADDER = [10, 20, None]  # None = all transactions

        if avg_price_years:
            windows = [avg_price_years] + [
                w for w in EXPANSION_LADDER if w is None or w > avg_price_years
            ]
        else:
            windows = [None]

        recent_transactions = []
        chosen_window = None
        for w in windows:
            if w is None:
                recent_transactions = transactions
                chosen_window = None
            else:
                cutoff_date = (datetime.now() - timedelta(days=w * 365)).strftime("%Y-%m-%d")
                recent_transactions = [t for t in transactions if t.get("date", "") >= cutoff_date]
                chosen_window = w
            if len(recent_transactions) >= MIN_AVG_SAMPLE or w is None:
                break

        window_label = f"{chosen_window} most recent years" if chosen_window is not None else "all years"
        print(f"  📊 Average-price window: {window_label}, {len(recent_transactions)} txns (min={MIN_AVG_SAMPLE})")

        prices = [t["price"] for t in recent_transactions if t["price"] > 0]

        if not prices:
            # Still empty (shouldn't happen, since the last step is all data): fall back
            prices = [t["price"] for t in transactions if t["price"] > 0]
            recent_transactions = transactions
            chosen_window = None

        if not prices:
            return {
                "postcode": postcode,
                "error": "No valid price data",
                "summary": None
            }

        # Basic statistics
        avg_price = sum(prices) / len(prices)
        sorted_prices = sorted(prices)
        median_price = sorted_prices[len(sorted_prices) // 2]
        min_price = min(prices)
        max_price = max(prices)

        # Price per m² (if floor-area data is available)
        prices_per_sqm = [t.get("price_per_sqm", 0) for t in recent_transactions
                         if t.get("price_per_sqm", 0) > 0]
        # Sample < MIN_SQM_SAMPLE → no figure: £/m² from a single-digit number of matches is too noisy; better missing than wrong
        price_per_sqm_count = len(prices_per_sqm)
        if price_per_sqm_count < self.MIN_SQM_SAMPLE:
            avg_price_per_sqm = 0
        else:
            avg_price_per_sqm = sum(prices_per_sqm) / price_per_sqm_count

        # 5. Statistics by property type
        by_property_type = defaultdict(list)
        for t in transactions:
            ptype = t.get("property_type", "O")
            by_property_type[ptype].append(t)

        property_type_stats = {}
        for ptype, type_transactions in by_property_type.items():
            type_prices = [t["price"] for t in type_transactions]
            # Average £/m² for this type
            type_sqm_prices = [t.get("price_per_sqm", 0) for t in type_transactions
                              if t.get("price_per_sqm", 0) > 0]
            avg_sqm = round(sum(type_sqm_prices) / len(type_sqm_prices)) if type_sqm_prices else None

            property_type_stats[ptype] = {
                "name": self.PROPERTY_TYPE_NAMES.get(ptype, ptype),
                "count": len(type_prices),
                "average_price": round(sum(type_prices) / len(type_prices)),
                "median_price": sorted(type_prices)[len(type_prices) // 2],
                "min_price": min(type_prices),
                "max_price": max(type_prices),
                "price_per_sqm": avg_sqm
            }

        # 6. Yearly statistics (including annualised growth from repeat sales)
        by_year = defaultdict(list)
        for t in transactions:
            try:
                year = int(t["date"][:4])
                by_year[year].append(t)
            except (ValueError, TypeError, IndexError):
                continue

        # Compute repeat sales first (used in the yearly stats)
        repeat_sales = self._identify_repeat_sales(pairing_transactions)

        # Group repeat-sale annualised growth by year
        repeat_by_year = defaultdict(list)
        for rs in repeat_sales:
            try:
                sale_year = int(rs["current_sale"]["date"][:4])
                repeat_by_year[sale_year].append(rs["annualized_return"])
            except (ValueError, TypeError, IndexError):
                continue

        yearly_stats = []
        sorted_years = sorted(by_year.keys())
        for year in sorted_years:
            year_transactions = by_year[year]
            year_prices = [t["price"] for t in year_transactions]
            current_avg = sum(year_prices) / len(year_prices)
            current_count = len(year_prices)

            # Average £/m² for the year
            year_sqm_prices = [t.get("price_per_sqm", 0) for t in year_transactions
                              if t.get("price_per_sqm", 0) > 0]
            avg_sqm = round(sum(year_sqm_prices) / len(year_sqm_prices)) if year_sqm_prices else None

            # Annualised growth of the year's repeat sales
            year_repeat_returns = repeat_by_year.get(year, [])
            if year_repeat_returns:
                avg_annualized_return = sum(year_repeat_returns) / len(year_repeat_returns)
                repeat_count = len(year_repeat_returns)
            else:
                avg_annualized_return = None
                repeat_count = 0

            yearly_stats.append({
                "year": year,
                "count": current_count,
                "average_price": round(current_avg),
                "median_price": sorted(year_prices)[len(year_prices) // 2],
                "price_per_sqm": avg_sqm,
                "repeat_sales_count": repeat_count,  # number of repeat sales that year
                "avg_annualized_return": round(avg_annualized_return, 2) if avg_annualized_return is not None else None
            })

        # 7. Repeat-sale highlights (reusing the pairs from step 6)
        # Find examples with especially large gains or losses
        notable_gains = []
        notable_losses = []
        if repeat_sales:
            # Sort by gain to find the biggest gains
            sorted_by_gain = sorted(repeat_sales, key=lambda x: x["price_change_pct"], reverse=True)
            # A gain > 30% or annualised > 10% counts as a notable gain
            notable_gains = [rs for rs in sorted_by_gain
                           if rs["price_change_pct"] > 30 or rs["annualized_return"] > 10][:3]

            # Find the biggest losses (negative change)
            sorted_by_loss = sorted(repeat_sales, key=lambda x: x["price_change_pct"])
            # A loss > 10% counts as a notable loss
            notable_losses = [rs for rs in sorted_by_loss if rs["price_change_pct"] < -10][:3]

        # 8. Compute trends (based on repeat-sale data)
        trends = self._calculate_price_trends(transactions, years, repeat_sales)

        # 9. Target-price comparison
        target_comparison = None
        if target_price:
            percentile = sum(1 for p in prices if p <= target_price) / len(prices) * 100
            target_comparison = {
                "target_price": target_price,
                "vs_average": round((target_price / avg_price - 1) * 100, 2),
                "vs_median": round((target_price / median_price - 1) * 100, 2),
                "percentile": round(percentile, 1),
                "assessment": self._assess_target_price(target_price, avg_price, median_price)
            }

        # Build the result
        analysis = {
            "postcode": postcode,
            "area_scope": geo_scope.get("scope", "unit"),
            "area_scope_postcode": geo_scope.get("postcode", postcode),
            "analysis_date": datetime.now().isoformat(),
            "years_analyzed": years,
            "avg_price_years": chosen_window,
            "avg_price_years_requested": avg_price_years,
            "summary": {
                "total_transactions": len(transactions),
                "avg_price_transactions": len(recent_transactions),
                "average_price": round(avg_price),
                "median_price": median_price,
                "min_price": min_price,
                "max_price": max_price,
                "price_per_sqm": round(avg_price_per_sqm) if avg_price_per_sqm else None,
                "price_per_sqm_count": price_per_sqm_count
            },
            "by_property_type": property_type_stats,
            "yearly_stats": yearly_stats,
            "repeat_sales": self._select_repeat_sales(repeat_sales),
            "repeat_sales_count": len(repeat_sales),
            # Std dev of annualised returns over all pairs (used for the "growth stability" score; the displayed sample is biased, so it cannot be used)
            "repeat_sales_return_std": (
                round(statistics.stdev([rs["annualized_return"] for rs in repeat_sales]), 2)
                if len(repeat_sales) >= 2 else None
            ),
            "notable_gains": notable_gains,  # biggest-gain examples
            "notable_losses": notable_losses,  # biggest-loss examples
            "trends": trends,
            "target_comparison": target_comparison,
            "transactions": transactions[:20]  # return only the 20 most recent
        }

        # Print the report
        self._print_analysis_report(analysis)

        return analysis

    def _assess_target_price(
        self,
        target: int,
        average: float,
        median: int
    ) -> str:
        """Assess the target price"""
        diff_avg = (target / average - 1) * 100
        diff_med = (target / median - 1) * 100

        if diff_avg < -15 and diff_med < -15:
            return "Well below market price, worth a look"
        elif diff_avg < -5 and diff_med < -5:
            return "Slightly below market price"
        elif diff_avg > 15 and diff_med > 15:
            return "Well above market price, proceed with caution"
        elif diff_avg > 5 and diff_med > 5:
            return "Slightly above market price"
        else:
            return "Close to the market average"

    def _print_analysis_report(self, analysis: Dict[str, Any]) -> None:
        """Print the analysis report"""
        summary = analysis.get("summary", {})
        trends = analysis.get("trends", {})
        target = analysis.get("target_comparison")

        print(f"\n{'='*70}")
        print(f"📊 Analysis report: {analysis['postcode']}")
        print(f"{'='*70}")

        # Basic statistics
        print(f"\n📌 Basic statistics ({summary.get('total_transactions', 0)} transactions)")
        print(f"   Average price: £{summary.get('average_price', 0):,}")
        print(f"   Median price: £{summary.get('median_price', 0):,}")
        print(f"   Price range: £{summary.get('min_price', 0):,} - £{summary.get('max_price', 0):,}")
        if summary.get('price_per_sqm'):
            print(f"   Average per m²: £{summary.get('price_per_sqm'):,}/m²")

        # Trend
        print(f"\n📈 Price trend")
        direction_emoji = {"up": "↗️ Rising", "down": "↘️ Falling", "stable": "➡️ Stable"}.get(
            trends.get("direction", "unknown"), "❓ Unknown"
        )
        print(f"   Trend direction: {direction_emoji}")

        # Show annualised returns based on repeat sales
        sample_count = trends.get("sample_count", 0)
        if sample_count > 0:
            print(f"   (based on {sample_count} repeat sales)")
            if trends.get("10_year_cagr") is not None:
                count_10 = trends.get("10_year_count", 0)
                print(f"   10-year avg annualised: {trends['10_year_cagr']:+.2f}%/yr ({count_10} txns)")
            if trends.get("5_year_cagr") is not None:
                count_5 = trends.get("5_year_count", 0)
                print(f"   5-year avg annualised: {trends['5_year_cagr']:+.2f}%/yr ({count_5} txns)")
            if trends.get("3_year_cagr") is not None:
                count_3 = trends.get("3_year_count", 0)
                print(f"   3-year avg annualised: {trends['3_year_cagr']:+.2f}%/yr ({count_3} txns)")
        else:
            print(f"   ⚠️  No repeat-sale data; cannot compute annualised returns")

        # By property type
        by_type = analysis.get("by_property_type", {})
        if by_type:
            print(f"\n🏠 By property type")
            for ptype, stats in sorted(by_type.items(), key=lambda x: x[1]["count"], reverse=True):
                sqm_str = f", £{stats['price_per_sqm']:,}/m²" if stats.get('price_per_sqm') else ""
                print(f"   {stats['name']}: {stats['count']} txns, avg £{stats['average_price']:,}{sqm_str}")

        # Repeat sales
        repeat_count = analysis.get("repeat_sales_count", 0)
        notable_gains = analysis.get("notable_gains", [])
        notable_losses = analysis.get("notable_losses", [])

        if repeat_count > 0:
            print(f"\n🔄 Repeat-sale analysis ({repeat_count} cases)")

            # Show the biggest-gain examples
            if notable_gains:
                print(f"\n   📈 Biggest gains:")
                for rs in notable_gains:
                    print(f"   {rs['address'][:35]}...")
                    print(f"      {rs['previous_sale']['date'][:10]}: £{rs['previous_sale']['price']:,}")
                    print(f"      {rs['current_sale']['date'][:10]}: £{rs['current_sale']['price']:,}")
                    change_str = f"+{rs['price_change_pct']:.1f}%"
                    print(f"      🚀 Gain: {change_str} (annualised +{rs['annualized_return']:.1f}%, held {rs['years_held']:.1f} yrs)")

            # Show the biggest-loss examples
            if notable_losses:
                print(f"\n   📉 Biggest losses:")
                for rs in notable_losses:
                    print(f"   {rs['address'][:35]}...")
                    print(f"      {rs['previous_sale']['date'][:10]}: £{rs['previous_sale']['price']:,}")
                    print(f"      {rs['current_sale']['date'][:10]}: £{rs['current_sale']['price']:,}")
                    change_str = f"{rs['price_change_pct']:.1f}%"
                    print(f"      ⚠️  Loss: {change_str} (annualised {rs['annualized_return']:.1f}%, held {rs['years_held']:.1f} yrs)")

            # If there are no notable gains/losses, show ordinary repeat-sale examples
            if not notable_gains and not notable_losses:
                print(f"\n   Typical cases:")
                for rs in analysis.get("repeat_sales", [])[:5]:
                    print(f"   {rs['address'][:35]}...")
                    print(f"      {rs['previous_sale']['date'][:10]}: £{rs['previous_sale']['price']:,}")
                    print(f"      {rs['current_sale']['date'][:10]}: £{rs['current_sale']['price']:,}")
                    change_str = f"+{rs['price_change_pct']:.1f}%" if rs['price_change_pct'] > 0 else f"{rs['price_change_pct']:.1f}%"
                    print(f"      Change: {change_str} (annualised {rs['annualized_return']:.1f}%, held {rs['years_held']:.1f} yrs)")

        # Target-price comparison
        if target:
            print(f"\n🎯 Target-price analysis: £{target['target_price']:,}")
            vs_avg = target['vs_average']
            vs_med = target['vs_median']
            vs_avg_str = f"+{vs_avg:.1f}%" if vs_avg > 0 else f"{vs_avg:.1f}%"
            vs_med_str = f"+{vs_med:.1f}%" if vs_med > 0 else f"{vs_med:.1f}%"
            print(f"   vs average: {vs_avg_str}")
            print(f"   vs median: {vs_med_str}")
            print(f"   Price rank: percentile {target['percentile']:.0f} / 100")
            print(f"   Assessment: {target['assessment']}")

        # Yearly statistics (show only the last 10 years)
        yearly = analysis.get("yearly_stats", [])
        if yearly:
            # Check whether any £/m² data exists
            has_sqm_data = any(ys.get('price_per_sqm') for ys in yearly)

            # Show only the last 10 years
            current_year = datetime.now().year
            display_years = 10
            yearly_display = [ys for ys in yearly if ys['year'] >= current_year - display_years]

            total_years = analysis.get("years_analyzed", 30)
            print(f"\n📅 Yearly statistics (showing last {display_years} years of {total_years} analysed)")
            if has_sqm_data:
                print(f"   {'Year':<6} {'Txns':<6} {'Avg price':<12} {'Per m²':<10} {'Resale change':<18}")
            else:
                print(f"   {'Year':<6} {'Txns':<6} {'Avg price':<14} {'Resale change':<20}")
            print(f"   {'-'*60}")

            for ys in yearly_display:
                count = ys['count']
                avg_return = ys.get('avg_annualized_return')
                repeat_count = ys.get('repeat_sales_count', 0)
                sqm_price = ys.get('price_per_sqm')

                # Annualised resale growth
                if avg_return is not None and repeat_count > 0:
                    return_str = f"+{avg_return:.1f}%" if avg_return > 0 else f"{avg_return:.1f}%"
                    return_emoji = "📈" if avg_return > 5 else ("📉" if avg_return < -5 else "➡️")
                    return_info = f"{return_emoji} {return_str}/yr ({repeat_count} txns)"
                else:
                    return_info = "(no resale data)"

                # Output format
                if has_sqm_data:
                    sqm_str = f"£{sqm_price:,}/m²" if sqm_price else "-"
                    print(f"   {ys['year']:<6} {count:>2} txns £{ys['average_price']:>8,}   {sqm_str:<10} {return_info}")
                else:
                    print(f"   {ys['year']:<6} {count:>2} txns £{ys['average_price']:>9,}    {return_info}")

        print(f"\n{'='*70}\n")
