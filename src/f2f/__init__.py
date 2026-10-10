"""f2f: face-to-face contact networks, from characterisation to contagion control.

Package layout (one module per implementation stream):

- datasets     stream 1: load the SocioPatterns contact lists into one contact table
- views        stream 1: snapshot sequence, aggregated weighted graph, per-day splits
- cache        stream 1: the shared result-cache row format
- static_metrics  stream 2: lecture-1 metrics, degree-distribution fits, centralities
- nulls        stream 2: configuration model, degree-preserving rewiring, z-scores
- temporal_metrics, temporal_nulls, prediction  stream 3
"""

from . import (  # noqa: F401
    cache,
    datasets,
    nulls,
    prediction,
    static_metrics,
    temporal_metrics,
    temporal_nulls,
    views,
)

__all__ = [
    "datasets", "views", "cache", "static_metrics", "nulls",
    "temporal_metrics", "temporal_nulls", "prediction",
]
