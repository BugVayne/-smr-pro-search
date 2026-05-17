"""Pydantic data models shared across all services.

These describe the JSON payloads exchanged between microservices.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- Domain entities ---------------------------------------------------------

class Rascenka(BaseModel):
    """A single normative position (расценка / material / mechanism)."""
    obosn: str  # шифр, e.g. "ЕР-46-01-008"
    naim: str   # наименование
    tip: str = "100"
    ed_izm: Optional[str] = None  # единица измерения
    entity_type: str = "rate"     # 'rate' | 'material' | 'mechanism' | 'device'
    rate_group_id: Optional[int] = None
    rate_group_name: Optional[str] = None


class PTMMatchInfo(BaseModel):
    """A PTM that matched the query, with its top расценки."""
    ptm_naim: str
    overlap_score: float
    top_rascenki: List[Rascenka] = Field(default_factory=list)


class PTMGroup(BaseModel):
    """Result group for aggregated queries: a matched PTM with its typical расценки."""
    ptm_naim: str
    overlap_score: float
    members: List[Rascenka] = Field(default_factory=list)


class PTMTransaction(BaseModel):
    """A project-technical module taken from a historical estimate."""
    ptm_kod: str           # e.g. "Ж2-10"
    ptm_naim: str
    glava: Optional[str] = None
    items: List[str]       # list of rascenka obosn codes


# --- Service request / response payloads -------------------------------------

class PreprocessRequest(BaseModel):
    query: str


class PreprocessResponse(BaseModel):
    original: str
    corrected: str            # after SymSpell
    lemmas: List[str]         # normalised tokens
    expanded_terms: List[str] # synonyms / nearest neighbours
    query_type: str           # "atomic" | "aggregated"
    extracted_code: Optional[str] = None  # шифр расценки если обнаружен в запросе


class RetrieveRequest(BaseModel):
    corrected: str
    lemmas: List[str]
    expanded_terms: List[str]
    top_k: int = 100


class CandidateScore(BaseModel):
    obosn: str
    naim: str
    entity_type: str = "rate"
    rate_group_id: Optional[int] = None
    rate_group_name: Optional[str] = None
    bm25_score: float = 0.0
    sbert_score: float = 0.0
    ptm_score: float = 0.0
    rrf_score: float = 0.0
    bm25_rank: Optional[int] = None
    sbert_rank: Optional[int] = None
    ptm_rank: Optional[int] = None


class RetrieveResponse(BaseModel):
    candidates: List[CandidateScore]
    matched_ptms: List[PTMMatchInfo] = Field(default_factory=list)


class RankRequest(BaseModel):
    query: str
    lemmas: List[str]
    candidates: List[CandidateScore]
    context: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 20


class RankedItem(BaseModel):
    obosn: str
    naim: str
    score: float
    entity_type: str = "rate"
    rate_group_id: Optional[int] = None
    rate_group_name: Optional[str] = None
    features: Dict[str, float] = Field(default_factory=dict)
    related: List[Rascenka] = Field(default_factory=list)  # populated for atomic queries


class RankResponse(BaseModel):
    items: List[RankedItem]


class PostprocessRequest(BaseModel):
    items: List[RankedItem]
    query_type: str = "atomic"
    matched_ptms: List[PTMMatchInfo] = Field(default_factory=list)


class TechKit(BaseModel):
    """A recommended technological set of rascenki."""
    seed_obosn: str         # код, для которого построен комплект
    seed_naim: str
    members: List[Rascenka]
    confidence: float
    support: float = 0.0


class PostprocessResponse(BaseModel):
    items: List[RankedItem]
    kits: List[TechKit] = Field(default_factory=list)
    ptm_groups: List[PTMGroup] = Field(default_factory=list)


# --- Gateway-level pipeline --------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    corrected: str
    query_type: str
    items: List[RankedItem]
    kits: List[TechKit] = Field(default_factory=list)
    ptm_groups: List[PTMGroup] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    debug: Dict[str, Any] = Field(default_factory=dict)
