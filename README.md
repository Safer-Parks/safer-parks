# safer-parks

Safer Parks analysis, example notebooks, and Python library for geospatial analysis of park safety.

Please see the folder [Development notebooks](development_notebooks) to see an example workflow.

## Features

This package provides utilities for:
- Subsetting geospatial data to specific local authority districts
- Merging touching or intersecting park polygons to create unified park boundaries
- Cleaning and deduplicating attribute data
- Calculating park entrances based on OSM tags and intersecting paths

## Installation

Install in development mode:
```bash
pip install -e .
```

Or with test dependencies:
```bash
pip install -e ".[test]"
```

## Usage

```python
from safer_parks import (
    subset_to_LAD,
    merge_touching_or_intersecting_polygons,
    clean_and_deduplicate
)

# Clean messy attribute strings
clean_string = clean_and_deduplicate("park; playground; park; sports field")

# Merge touching park polygons
merged_parks = merge_touching_or_intersecting_polygons(parks_gdf)

# Subset parks to a specific local authority
local_parks = subset_to_LAD(lad_gdf, 'LAD_name', 'Bradford', parks_gdf)
```

## Functions

### `subset_to_LAD(LAD_gdf, LAD_column_name, LAD_name, data_to_subset)`
Subset greenspace data to only include features within a chosen local authority district.

### `merge_touching_or_intersecting_polygons(gdf)`
Combine greenspace geometries that are touching or intersecting to create unified park boundaries.

### `merge_touching_or_intersecting_polygons_condense(gdf)`
Iteratively merge touching polygons until no further merging is possible, with cleaned attributes.

### `clean_and_deduplicate(values, separator=';')`
Clean and deduplicate separator-separated strings, removing duplicates and excess whitespace.

## Testing

Run tests with:
```bash
pytest tests/
```
