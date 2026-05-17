"""Building technological kits and related-item enrichment.

build_kits: FP-Growth association rules → TechKit objects (aggregated mode).
enrich_with_related: per-item co-occurring расценки enrichment (atomic mode).
build_ptm_groups: matched PTMs → PTMGroup objects (aggregated mode).
"""
from __future__ import annotations

from typing import List

from common.db import find_rules_for, get_cooccurring_from_transactions, get_rascenki
from common.models import PTMGroup, PTMMatchInfo, RankedItem, Rascenka, TechKit


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


def enrich_with_related(items: List[RankedItem], max_related: int = 3) -> List[RankedItem]:
    """Atomic mode: attach co-occurring расценки to each ranked item.

    Uses FP-Growth consequents (confidence-sorted) as primary source.
    Falls back to raw transaction co-occurrence when no rules exist.
    """
    for item in items:
        if item.entity_type != "rate":
            continue

        rules = find_rules_for(item.obosn, limit=15)
        if rules:
            consequents: dict[str, float] = {}
            for rule in rules:
                for c in rule["consequent"]:
                    if c == item.obosn:
                        continue
                    if c not in consequents or rule["confidence"] > consequents[c]:
                        consequents[c] = rule["confidence"]
            related_codes = sorted(consequents, key=lambda x: -consequents[x])[:max_related]
        else:
            cooc = get_cooccurring_from_transactions(item.obosn, limit=max_related)
            related_codes = [r["obosn"] for r in cooc]

        if related_codes:
            rows = get_rascenki(related_codes)
            rows_by_obosn = {r["obosn"]: r for r in rows}
            item.related = [
                Rascenka(
                    obosn=r["obosn"],
                    naim=r["naim"],
                    tip=r.get("tip") or "100",
                    ed_izm=r.get("ed_izm"),
                    entity_type=r.get("entity_type", "rate"),
                    rate_group_id=r.get("rate_group_id"),
                )
                for code in related_codes
                if (r := rows_by_obosn.get(code))
            ]

    return items


def build_ptm_groups(matched_ptms: List[PTMMatchInfo], max_members: int = 15) -> List[PTMGroup]:
    """Aggregated mode: convert PTMMatchInfo list to PTMGroup list."""
    return [
        PTMGroup(
            ptm_naim=ptm.ptm_naim,
            overlap_score=ptm.overlap_score,
            members=ptm.top_rascenki[:max_members],
        )
        for ptm in matched_ptms
        if ptm.top_rascenki
    ]
