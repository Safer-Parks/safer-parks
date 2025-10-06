"""
Safer Parks: A package for geospatial analysis of park safety and accessibility.
"""

from .geospatial_utils import (
    subset_to_LAD,
    merge_touching_or_intersecting_polygons,
    merge_touching_or_intersecting_polygons_condense,
    clean_and_deduplicate,
    fuzzy_match_score,
    match_parks_to_greenspace
)

from .entrances import (
    extract_park_entrances
)

__version__ = "0.1.0"

__all__ = [
    "subset_to_LAD",
    "merge_touching_or_intersecting_polygons", 
    "merge_touching_or_intersecting_polygons_condense",
    "clean_and_deduplicate",
    "fuzzy_match_score",
    "match_parks_to_greenspace",
    "extract_park_entrances"
]