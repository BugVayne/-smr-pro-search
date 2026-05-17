"""Fine-tune SBERT on domain pairs.

Positive pairs:
    (ptm.naim, rascenka.naim) for every rascenka inside that PTM.

Hard negatives:
    rascenki from other PTMs with the *same* ptm_kod but not from this
    transaction.

Trained with MultipleNegativesRankingLoss (in-batch negatives).
"""
from __future__ import annotations

import random
from typing import List

import os

from common.config import SBERT_BASE_MODEL, SBERT_MODEL_DIR
from common.db import all_transactions, get_rascenki
from common.gpu import get_device, gpu_batch_size
from common.logging_config import get_logger

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

log = get_logger(__name__)


def _build_pairs(min_pairs: int = 5) -> List[tuple]:
    transactions = all_transactions()
    if not transactions:
        return []

    pairs_seen: set[tuple[str, str]] = set()
    pairs: List[tuple] = []

    for tx in transactions:
        if not tx["ptm_naim"]:
            continue
        rows = get_rascenki(tx["items"])
        name_map = {r["obosn"]: r["naim"] for r in rows}
        for obosn in tx["items"]:
            naim = name_map.get(obosn)
            if not naim:
                continue
            pair = (tx["ptm_naim"], naim)
            # Deduplicate: the same (ptm_naim, rascenka_naim) pair from 100
            # projects is one model signal, not 100 — repeating it would bias
            # training toward common PTM names.
            if pair not in pairs_seen:
                pairs_seen.add(pair)
                pairs.append(pair)

    log.info("Built %d unique positive pairs (%d transactions processed)",
             len(pairs), len(transactions))
    return pairs if len(pairs) >= min_pairs else []


def train(epochs: int = 1, batch_size: int = None) -> bool:
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        log.warning("sentence-transformers not installed – skipping SBERT fine-tune")
        return False

    device = get_device()
    if batch_size is None:
        # RTX 3050 Ti (4 GB) fits ~8 with rubert-base + Adam optimizer state.
        # Larger batches OOM. MultipleNegativesRankingLoss uses in-batch
        # negatives so batch=8 still gives 7 negatives per anchor.
        batch_size = gpu_batch_size(cpu_size=16, gpu_size=8)

    pairs = _build_pairs()
    if not pairs:
        log.warning("Not enough pairs – skipping SBERT fine-tune")
        return False

    examples = [InputExample(texts=[a, b]) for a, b in pairs]
    random.shuffle(examples)

    log.info("Loading base model: %s on %s", SBERT_BASE_MODEL, device.upper())
    try:
        model = SentenceTransformer(SBERT_BASE_MODEL, device=device)
    except Exception as exc:
        log.warning("Could not load base SBERT (%s) – skipping", exc)
        return False

    loader = DataLoader(examples, batch_size=batch_size, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    use_amp = (device == "cuda")
    log.info("Fine-tuning SBERT for %d epoch(s) on %d examples (batch=%d, amp=%s)",
             epochs, len(examples), batch_size, use_amp)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=max(1, len(loader) // 10),
        show_progress_bar=True,
        use_amp=use_amp,
    )
    SBERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(SBERT_MODEL_DIR))
    log.info("Saved fine-tuned SBERT to %s", SBERT_MODEL_DIR)
    return True


if __name__ == "__main__":
    train()
