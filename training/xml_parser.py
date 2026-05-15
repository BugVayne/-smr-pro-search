"""Parser for SMR-Pro XML estimates.

Designed to scale to many large XML files dropped into ``data/raw/``:

* ``parse_xml`` uses :func:`lxml.etree.iterparse` to stream the file —
  PTMs are yielded one by one and processed nodes are cleared so peak
  memory stays bounded regardless of file size.
* ``parse_directory`` scans recursively (``**/*.xml``), so you can drop
  per-year or per-project subfolders into ``data/raw/`` without
  reorganising anything.

Expected hierarchy (see Документ 3 of the diploma):

    <stroyka>
      <ob_smeta glava="...">
        <loc_smeta>
          <ptm kod="..." naim="...">
            <rascenka obosn="..." naim="..." tip="100|101|103|...">
              <materialy><resurs .../></materialy>
              <mehanizmy><resurs .../></mehanizmy>
            </rascenka>
          </ptm>
        </loc_smeta>
      </ob_smeta>
    </stroyka>

Only rascenki with ``tip`` ∈ {100, 101, 103} are kept for transactions.
Service tips (104, 106, 149, …) are ignored.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from lxml import etree

from common.config import ALLOWED_TIP
from common.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class ParsedRascenka:
    obosn: str
    naim: str
    tip: str
    ed_izm: Optional[str] = None


@dataclass
class ParsedPTM:
    ptm_kod: str
    ptm_naim: str
    glava: Optional[str]
    rascenki: List[ParsedRascenka]


# ---------------------------------------------------------------------------
# Streaming parser for a single file
# ---------------------------------------------------------------------------

def parse_xml(xml_path: Path) -> Iterator[ParsedPTM]:
    """Yield one :class:`ParsedPTM` per ``<ptm>`` element.

    Uses iterative parsing so memory usage does not grow with file size.
    ``<ob_smeta glava="...">`` attributes are tracked via a small stack
    so each yielded PTM carries its containing chapter.
    """
    glava_stack: List[Optional[str]] = []

    context = etree.iterparse(str(xml_path), events=("start", "end"))
    for event, elem in context:
        if event == "start":
            if elem.tag == "ob_smeta":
                glava_stack.append(elem.get("glava"))
            continue

        # event == "end"
        if elem.tag == "ptm":
            yield _ptm_from_element(elem, glava_stack[-1] if glava_stack else None)
            # Free memory: drop the processed subtree.
            elem.clear()
            # Also drop previous sibling PTMs that have already been
            # processed, otherwise the parent loc_smeta accumulates them
            # all in memory.
            parent = elem.getparent()
            if parent is not None:
                prev = elem.getprevious()
                while prev is not None:
                    del parent[0]
                    prev = elem.getprevious()
        elif elem.tag == "ob_smeta" and glava_stack:
            glava_stack.pop()


def _ptm_from_element(elem, glava: Optional[str]) -> ParsedPTM:
    rascenki: List[ParsedRascenka] = []
    for r in elem.iter("rascenka"):
        tip = r.get("tip", "")
        if tip not in ALLOWED_TIP:
            continue
        obosn = r.get("obosn", "")
        if not obosn:
            continue
        rascenki.append(
            ParsedRascenka(
                obosn=obosn,
                naim=r.get("naim", ""),
                tip=tip,
                ed_izm=r.get("ed_izm"),
            )
        )
    return ParsedPTM(
        ptm_kod=elem.get("kod", ""),
        ptm_naim=elem.get("naim", ""),
        glava=glava,
        rascenki=rascenki,
    )


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

def discover_xml_files(xml_dir: Path, recursive: bool = True) -> List[Path]:
    """Find all XML files under ``xml_dir``.

    Recursive by default so the user can organise files into subfolders
    (e.g. ``data/raw/2024/...``). Sorted for reproducibility.
    """
    if not xml_dir.exists():
        return []
    pattern = "**/*.xml" if recursive else "*.xml"
    return sorted(xml_dir.glob(pattern))


def parse_directory(
    xml_dir: Path, recursive: bool = True
) -> Iterator[Tuple[Path, ParsedPTM]]:
    """Iterate (xml_path, ptm) pairs across every XML file in the directory.

    Each PTM is yielded as soon as it's parsed — caller can keep
    streaming and never holds the whole dataset in memory.
    """
    files = discover_xml_files(xml_dir, recursive=recursive)
    log.info("Discovered %d XML file(s) under %s", len(files), xml_dir)
    for i, f in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), f.relative_to(xml_dir) if f.is_relative_to(xml_dir) else f)
        try:
            n_ptms = 0
            for ptm in parse_xml(f):
                n_ptms += 1
                yield f, ptm
            log.info("        → %d PTMs", n_ptms)
        except Exception as exc:
            log.warning("        ✗ failed: %s", exc)


def transaction_hash(items: List[str]) -> str:
    return hashlib.sha1("|".join(sorted(items)).encode("utf-8")).hexdigest()


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the file content (streamed in 1 MB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
