"""
Example usage of safer_parks geospatial utilities.
"""

import geopandas as gpd
from shapely.geometry import Polygon
from safer_parks import (
    subset_to_LAD,
    merge_touching_or_intersecting_polygons,
    clean_and_deduplicate
)


def example_usage():
    """Demonstrate the usage of safer_parks functions."""
    
    # Example 1: Clean and deduplicate strings
    messy_string = "park; playground; park; sports field"
    clean_string = clean_and_deduplicate(messy_string)
    print(f"Original: {messy_string}")
    print(f"Cleaned: {clean_string}")
    
    # Example 2: Create sample geospatial data
    # Two touching polygons representing park areas
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])  # Touches poly1
    
    parks_gdf = gpd.GeoDataFrame({
        'name': ['Sports Field', 'Playground'],
        'type': ['sports', 'playground'],
        'geometry': [poly1, poly2]
    }, crs='EPSG:4326')
    
    print(f"\nOriginal parks: {len(parks_gdf)} features")
    
    # Example 3: Merge touching polygons
    merged_parks = merge_touching_or_intersecting_polygons(parks_gdf)
    print(f"After merging: {len(merged_parks)} features")
    
    if len(merged_parks) > 0:
        merged_name = merged_parks.iloc[0]['name']
        print(f"Merged park name: {merged_name}")


if __name__ == "__main__":
    example_usage()