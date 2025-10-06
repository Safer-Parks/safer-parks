"""
Tests for geospatial utility functions.
"""

import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
from safer_parks.geospatial_utils import (
    subset_to_LAD,
    merge_touching_or_intersecting_polygons,
    merge_touching_or_intersecting_polygons_condense,
    clean_and_deduplicate
)


class TestCleanAndDeduplicate:
    """Tests for the clean_and_deduplicate function."""
    
    def test_basic_deduplication(self):
        """Test basic deduplication functionality."""
        result = clean_and_deduplicate("apple; banana; apple; cherry", separator=";")
        assert result == "apple;banana;cherry"
    
    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped."""
        result = clean_and_deduplicate(" apple ; banana ; apple ", separator=";")
        assert result == "apple;banana"
    
    def test_empty_values(self):
        """Test handling of empty values."""
        result = clean_and_deduplicate("apple;; banana; ; cherry", separator=";")
        assert result == "apple;banana;cherry"


class TestGeospatialFunctions:
    """Tests for geospatial functions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create simple test polygons
        self.poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.poly2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])  # Touches poly1
        self.poly3 = Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])  # Separate
        
        # Create test GeoDataFrames
        self.gdf_parks = gpd.GeoDataFrame({
            'name': ['Park A', 'Park B', 'Park C'],
            'type': ['playground', 'sports', 'nature'],
            'geometry': [self.poly1, self.poly2, self.poly3]
        }, crs='EPSG:4326')
        
        # Create LAD boundary that contains poly1 and poly2
        self.lad_boundary = Polygon([(-0.5, -0.5), (2.5, -0.5), (2.5, 1.5), (-0.5, 1.5)])
        self.gdf_lad = gpd.GeoDataFrame({
            'LAD_name': ['Test LAD'],
            'geometry': [self.lad_boundary]
        }, crs='EPSG:4326')
    
    def test_subset_to_LAD(self):
        """Test subsetting data to a local authority district."""
        result = subset_to_LAD(
            self.gdf_lad, 'LAD_name', 'Test LAD', self.gdf_parks
        )
        
        # Should contain Park A and Park B (within LAD), but not Park C
        assert len(result) == 2
        assert 'Park A' in result['name'].values
        assert 'Park B' in result['name'].values
        assert 'Park C' not in result['name'].values
    
    def test_merge_touching_polygons(self):
        """Test merging of touching polygons."""
        result = merge_touching_or_intersecting_polygons(self.gdf_parks)
        
        # Should merge Park A and Park B (they touch), Park C remains separate
        assert len(result) == 2
        
        # Check that attributes are combined properly
        merged_row = result[result['name'].str.contains('Park A')].iloc[0]
        assert 'Park A' in merged_row['name']
        assert 'Park B' in merged_row['name']


if __name__ == "__main__":
    pytest.main([__file__])
