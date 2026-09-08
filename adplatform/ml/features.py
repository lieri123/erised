# features.py — the ONLY place features are computed.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .embeddings import (
    EMBEDDING_BLOCK_SIZE,
    EMBEDDING_FEATURE_NAMES,
    NEUTRAL_EMBEDDING_BLOCK,
    EmbeddingTable,
)

# 3 added the learned embedding block. The base block below did not change and
# kept its indexes, which is what lets train_ctr.py go on training from v2
# impressions. See TRAINABLE_FEATURE_VERSIONS there.
FEATURE_VERSION = 3

# The hand-written block. Order is load-bearing: these indexes appear in logged
# vectors going back months, so nothing may be inserted in the middle.
BASE_FEATURE_NAMES: tuple[str, ...] = (
    "hour_of_day",            # 0-23
    "day_of_week",            # 0=Mon
    "is_weekend",
    "is_mobile",
    "is_desktop",
    "is_tablet",
    "keyword_overlap",        # raw count of shared keywords
    "keyword_overlap_ratio",  # overlap / len(ad keywords)
    "n_ad_keywords",
    "n_page_keywords",
    "target_cpm",
    "ad_ctr_prior",           # smoothed historical CTR for this ad
    "placement_ctr_prior",    # smoothed historical CTR for this placement
    "pair_ctr_prior",         # smoothed historical CTR for (ad, placement)
    "pair_impressions_log",   # log1p(impressions) — tells the model how much
                              # to trust pair_ctr_prior
    "ad_age_days",
    "budget_pacing",          # spent_today / daily_budget, 0.0-1.0+
)

N_BASE_FEATURES = len(BASE_FEATURE_NAMES)

# Appending rather than interleaving is what keeps a v2 vector a valid prefix
# of a v3 one.
FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + EMBEDDING_FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)

# Import-time tripwires for the two silent failures here: a neutral placeholder
# of the wrong width, and a total that no longer adds up. Either one shows up
# downstream as a model trained on shifted columns.
assert len(NEUTRAL_EMBEDDING_BLOCK) == EMBEDDING_BLOCK_SIZE
assert N_FEATURES == N_BASE_FEATURES + EMBEDDING_BLOCK_SIZE
 
 
@dataclass(frozen=True)
class RequestContext:
    """Everything about the bid request that is independent of which ad we score."""
 
    publisher_id: str
    placement_id: str
    device_type: str            
    page_keywords: tuple[str, ...]
    request_ts: datetime
 
    @classmethod
    def build(
        cls,
        publisher_id: str,
        placement_id: str,
        device_type: str,
        page_keywords: list[str],
        request_ts: datetime | None = None,
    ) -> "RequestContext":
        ts = request_ts or datetime.now(timezone.utc)
        kws = tuple(sorted({k.strip().lower() for k in page_keywords if k.strip()}))
        return cls(
            publisher_id=publisher_id,
            placement_id=placement_id,
            device_type=(device_type or "").strip().lower(),
            page_keywords=kws,
            request_ts=ts,
        )
 
 
@dataclass
class CtrStats:
    global_ctr: float = 0.010
    prior_strength: float = 200.0
    # key -> (impressions, clicks)
    ad_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    placement_counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    pair_counts: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)
 
    def _smoothed(self, counts: tuple[int, int] | None) -> float:
        alpha = self.global_ctr * self.prior_strength
        beta = (1.0 - self.global_ctr) * self.prior_strength
        if counts is None:
            return self.global_ctr
        impressions, clicks = counts
        return (clicks + alpha) / (impressions + alpha + beta)
 
    def ad_ctr(self, ad_id: str) -> float:
        return self._smoothed(self.ad_counts.get(ad_id))
 
    def placement_ctr(self, placement_id: str) -> float:
        return self._smoothed(self.placement_counts.get(placement_id))
 
    def pair_ctr(self, ad_id: str, placement_id: str) -> float:
        return self._smoothed(self.pair_counts.get((ad_id, placement_id)))
 
    def pair_impressions(self, ad_id: str, placement_id: str) -> int:
        return self.pair_counts.get((ad_id, placement_id), (0, 0))[0]
 
    @property
    def is_empty(self) -> bool:
        return not self.ad_counts and not self.placement_counts
 
 
EMPTY_STATS = CtrStats()
 
 
def extract_features(
    ad,
    ctx: RequestContext,
    stats: CtrStats = EMPTY_STATS,
    embeddings: EmbeddingTable | None = None,
) -> list[float]:
    """
    The full vector: the hand-written block, then the learned one.

    `embeddings` comes from the loaded artifact and has to be the table that
    shipped with the booster being scored. A different table, or None when the
    booster was trained with one, is train/serve skew that nothing downstream
    can see, since the columns are still plausible floats. ctr_model reads both
    from one _Artifact reference so they cannot be mismatched, and refuses an
    artifact whose metadata claims embeddings it did not ship.
    """
    ts = ctx.request_ts
    hour = float(ts.hour)
    dow = float(ts.weekday())
 
    ad_kws = {k.lower() for k in (ad.target_keywords or ())}
    overlap = len(ad_kws & set(ctx.page_keywords))
    overlap_ratio = overlap / len(ad_kws) if ad_kws else 0.0
 
    daily_budget = float(getattr(ad, "daily_budget_usd", 0.0) or 0.0)
    spent = float(getattr(ad, "spent_today_usd", 0.0) or 0.0)
    pacing = (spent / daily_budget) if daily_budget > 0 else 0.0
 
    created_at = getattr(ad, "created_at", None)
    if isinstance(created_at, datetime):
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (ts - created_at).total_seconds() / 86400.0)
    else:
        age_days = 0.0
 
    device = ctx.device_type
    pair_imps = stats.pair_impressions(ad.ad_id, ctx.placement_id)
 
    import math
 
    return [
        hour,
        dow,
        1.0 if dow >= 5 else 0.0,
        1.0 if device == "mobile" else 0.0,
        1.0 if device == "desktop" else 0.0,
        1.0 if device == "tablet" else 0.0,
        float(overlap),
        overlap_ratio,
        float(len(ad_kws)),
        float(len(ctx.page_keywords)),
        float(ad.target_cpm),
        stats.ad_ctr(ad.ad_id),
        stats.placement_ctr(ctx.placement_id),
        stats.pair_ctr(ad.ad_id, ctx.placement_id),
        math.log1p(pair_imps),
        age_days,
        pacing,
        *(
            embeddings.block(ad.ad_id, ctx.placement_id)
            if embeddings is not None
            else NEUTRAL_EMBEDDING_BLOCK
        ),
    ]
 
 
def features_to_dict(vec: list[float]) -> dict[str, float]:
    """For debugging and dashboards only. Never use this on the hot path."""
    return dict(zip(FEATURE_NAMES, vec))


def split_blocks(vec: list[float]) -> tuple[list[float], list[float]]:
    """The base and learned halves of a vector. Tests and inspection only."""
    return vec[:N_BASE_FEATURES], vec[N_BASE_FEATURES:]

