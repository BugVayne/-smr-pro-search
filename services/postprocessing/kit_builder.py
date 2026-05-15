"""Building technological kits from association rules.

For each of the top-N ranked items the kit builder queries the
association-rule store for rules whose antecedent contains that
rascenka. Consequents are aggregated, de-duplicated, and returned as
``TechKit`` objects.
"""
from __future__ import annotations

from typing import List

from common.db import find_rules_for, get_rascenki
from common.models import RankedItem, Rascenka, TechKit


def build_kits(items: List[RankedItem], top_seeds: int = 5, max_kit_size: int = 10) -> List[TechKit]:
    kits: List[TechKit] = []
    seen: set[tuple[str, str]] = set()  # (seed_obosn, member_obosn)

    # Only seed kits from compound rates – kits are technological groupings
    # of rates, not of individual materials or mechanisms.
    rate_items = [it for it in items if it.entity_type == "rate"]

    for seed in rate_items[:top_seeds]:
        rules = find_rules_for(seed.obosn, limit=20)
        if not rules:
            continue

        # Collect distinct consequent codes, weighted by best confidence seen
        consequents: dict[str, dict] = {}
        for rule in rules:
            for cons_obosn in rule["consequent"]:
                if cons_obosn == seed.obosn:
                    continue
                key = (seed.obosn, cons_obosn)
                if key in seen:
                    continue
                cur = consequents.get(cons_obosn)
                if cur is None or rule["confidence"] > cur["confidence"]:
                    consequents[cons_obosn] = {
                        "confidence": rule["confidence"],
                        "support": rule["support"],
                    }

        if not consequents:
            continue

        member_codes = list(consequents.keys())[:max_kit_size]
        members_rows = get_rascenki(member_codes)
        members = [
            Rascenka(
                obosn=r["obosn"],
                naim=r["naim"],
                tip=r.get("tip") or "100",
                ed_izm=r.get("ed_izm"),
                entity_type=r.get("entity_type", "rate"),
                rate_group_id=r.get("rate_group_id"),
            )
            for r in members_rows
        ]
        if not members:
            continue

        kit_conf = max(consequents[m.obosn]["confidence"] for m in members)
        kit_sup = max(consequents[m.obosn]["support"] for m in members)

        kits.append(
            TechKit(
                seed_obosn=seed.obosn,
                seed_naim=seed.naim,
                members=members,
                confidence=float(kit_conf),
                support=float(kit_sup),
            )
        )
        for m in members:
            seen.add((seed.obosn, m.obosn))

    return kits
