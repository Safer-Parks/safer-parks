"""
Geospatial utility functions for safer parks analysis.

This module contains functions for subsetting and merging geospatial data,
particularly useful for working with park and greenspace data.
"""

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from difflib import SequenceMatcher
import warnings


def subset_to_LAD(LAD_gdf, LAD_column_name, LAD_name, data_to_subset):
    """
    Subset a greenspace (or similar file) to only include greenspaces 
    within a chosen local authority.
    
    Parameters
    ----------
    LAD_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing local authority district boundaries
    LAD_column_name : str
        Name of the column containing LAD names
    LAD_name : str
        Name of the specific LAD to subset to
    data_to_subset : geopandas.GeoDataFrame
        Data to be subset to the chosen LAD
        
    Returns
    -------
    geopandas.GeoDataFrame
        Data subset to only include features within the chosen LAD
    """
    chosen_LAD = LAD_gdf.loc[LAD_gdf[LAD_column_name] == LAD_name, :]
    chosen_LAD = chosen_LAD.to_crs(data_to_subset.crs)
    data_subset_to_LAD = data_to_subset[data_to_subset.within(chosen_LAD.union_all())]
    return data_subset_to_LAD


def merge_touching_or_intersecting_polygons(gdf):
    """
    Combine greenspace geographies that are intersecting or touching 
    to provide a more simplified boundary for parks more closely aligned 
    to a perceived park.
    
    E.g. combines a park made up of woodland, sports fields and open 
    greenspaces into one park.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries to merge
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with touching or intersecting polygons merged
    """
    # Ensure the GeoDataFrame has a valid CRS
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    # Create a spatial index for efficient spatial queries
    spatial_index = gdf.sindex

    # Track which geometries have been processed
    merged = []
    used = set()

    for idx, geom in enumerate(gdf.geometry):
        if idx in used:
            continue

        # Find all geometries that intersect or touch the current one
        possible_matches_index = list(spatial_index.intersection(geom.bounds))
        candidates = gdf.iloc[possible_matches_index]
        touching_or_intersecting = candidates[candidates.geometry.apply(
            lambda x: x.intersects(geom) or x.touches(geom)
        )]

        # Combine all geometries
        merged_geom = unary_union(touching_or_intersecting.geometry)

        # Combine attributes by concatenating non-null values
        combined_attributes = {}
        for column in gdf.columns:
            if column != 'geometry':
                combined_attributes[column] = (
                    touching_or_intersecting[column]
                    .dropna()
                    .astype(str)
                    .str.cat(sep='; ')
                )

        # Append the merged geometry and attributes
        merged.append({**combined_attributes, 'geometry': merged_geom})

        # Mark these indices as used
        used.update(touching_or_intersecting.index)

    # Create a new GeoDataFrame with the merged results
    merged_gdf = gpd.GeoDataFrame(merged, crs=gdf.crs)

    return merged_gdf


def clean_and_deduplicate(values, separator=';'):
    """
    Clean a list of separator-separated strings by removing duplicates,
    stripping whitespace, and avoiding repeated separators.
    
    Parameters
    ----------
    values : str or list
        Values to clean and deduplicate
    separator : str, default ';'
        Separator character used to split and join values
        
    Returns
    -------
    str
        Cleaned and deduplicated string
    """
    items = [item.strip() for item in str(values).split(separator) if item.strip()]
    seen = set()
    cleaned = [x for x in items if not (x in seen or seen.add(x))]
    return separator.join(cleaned)


def merge_touching_or_intersecting_polygons_condense(gdf):
    """
    Iteratively combine touching or intersecting polygons until no further
    merging is possible. This is a more comprehensive version that continues
    merging until convergence.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries to merge
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with all touching or intersecting polygons merged,
        with cleaned and deduplicated attributes
    """
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    while True:
        spatial_index = gdf.sindex
        merged = []
        used = set()

        for idx, geom in enumerate(gdf.geometry):
            if idx in used:
                continue

            possible_matches_index = list(spatial_index.intersection(geom.bounds))
            candidates = gdf.iloc[possible_matches_index]
            touching_or_intersecting = candidates[candidates.geometry.apply(
                lambda x: x.intersects(geom) or x.touches(geom)
            )]

            merged_geom = unary_union(touching_or_intersecting.geometry)

            combined_attributes = {}
            for column in gdf.columns:
                if column != 'geometry':
                    raw_values = (
                        touching_or_intersecting[column]
                        .dropna()
                        .astype(str)
                        .str.cat(sep=', ')
                    )
                    combined_attributes[column] = clean_and_deduplicate(raw_values)

            merged.append({**combined_attributes, 'geometry': merged_geom})
            used.update(touching_or_intersecting.index)

        new_gdf = gpd.GeoDataFrame(merged, crs=gdf.crs)

        # Stop if no further reduction in number of geometries
        if len(new_gdf) == len(gdf):
            break

        gdf = new_gdf

    return gdf


def fuzzy_match_score(a, b):
    """
    Calculate fuzzy string matching score between two strings.
    
    Parameters
    ----------
    a : str
        First string to compare
    b : str  
        Second string to compare
        
    Returns
    -------
    float
        Similarity score between 0 and 1
    """
    if pd.isna(a) or pd.isna(b):
        return 0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def match_parks_to_greenspace(parks_gdf, greenspace_gdf, 
                              name_threshold=0.85, 
                              spatial_buffers=[0, 10, 15, 20, 25]):
    """
    Comprehensive park-to-greenspace matching using multiple strategies.
    
    Uses hierarchical matching strategies in order of confidence:
    1. Exact name matching (OS names)  
    2. Exact name matching (OSM names)
    3. Fuzzy name matching
    4. Spatial matching with increasing buffer sizes
    
    Parameters
    ----------
    parks_gdf : geopandas.GeoDataFrame
        GeoDataFrame of parks (points)
    greenspace_gdf : geopandas.GeoDataFrame
        GeoDataFrame of greenspace polygons
    name_threshold : float, default 0.85
        Minimum score for fuzzy name matching (0-1)
    spatial_buffers : list, default [0, 10, 15, 20, 25]
        List of buffer distances to try (in meters)
    
    Returns
    -------
    tuple
        (matched_parks, unmatched_parks, match_quality)
        - matched_parks: GeoDataFrame of successfully matched parks
        - unmatched_parks: GeoDataFrame of parks that couldn't be matched  
        - match_quality: DataFrame with quality metrics for each match
    """
    warnings.filterwarnings('ignore')
    
    results = []
    unmatched = parks_gdf.copy()
    match_log = []
    
    # Ensure parks have point geometry
    if 'geometry' not in unmatched.columns or unmatched.geometry.isna().any():
        unmatched = gpd.GeoDataFrame(
            unmatched.drop(columns=['geometry'], errors='ignore'),
            geometry=gpd.points_from_xy(unmatched.Longitude, unmatched.Latitude, crs="EPSG:4326")
        )
    
    print(f"Starting with {len(unmatched)} parks to match...")
    
    # Strategy 1: Exact name matching (OS names)
    print("\n=== Strategy 1: Exact OS Name Matching ===")
    if 'Name (OS)' in greenspace_gdf.columns:
        parks_temp = unmatched.rename(columns={'geometry': 'park_geometry'})
        greenspace_temp = greenspace_gdf.rename(columns={'geometry': 'greenspace_geometry'})
        
        exact_os_matches = parks_temp.merge(
            greenspace_temp, 
            left_on='Park Name', 
            right_on='Name (OS)', 
            how='inner'
        )
        
        if len(exact_os_matches) > 0:
            exact_os_matches = gpd.GeoDataFrame(
                exact_os_matches.drop(columns=['park_geometry']),
                geometry=exact_os_matches['greenspace_geometry']
            )
            exact_os_matches = exact_os_matches.drop(columns=['greenspace_geometry'])
            exact_os_matches['match_method'] = 'exact_name_os'
            exact_os_matches['match_quality'] = 1.0
            
            results.append(exact_os_matches)
            matched_indices = exact_os_matches['index'].tolist()
            unmatched = unmatched[~unmatched.index.isin(matched_indices)]
            print(f"Found {len(exact_os_matches)} exact OS name matches")
            
            for _, row in exact_os_matches.iterrows():
                match_log.append({
                    'park_name': row['Park Name'],
                    'matched_to': row['Name (OS)'],
                    'method': 'exact_name_os',
                    'match_quality': 1.0
                })
    
    # Strategy 2: Exact name matching (OSM names)
    print("\n=== Strategy 2: Exact OSM Name Matching ===")
    if 'Name (OSM)' in greenspace_gdf.columns and len(unmatched) > 0:
        # Temporarily rename geometry columns to avoid conflicts
        parks_temp = unmatched.rename(columns={'geometry': 'park_geometry'})
        greenspace_temp = greenspace_gdf.rename(columns={'geometry': 'greenspace_geometry'})
        
        exact_osm_matches = parks_temp.merge(
            greenspace_temp, 
            left_on='Park Name', 
            right_on='Name (OSM)', 
            how='inner'
        )
        
        if len(exact_osm_matches) > 0:
            exact_osm_matches = gpd.GeoDataFrame(
                exact_osm_matches.drop(columns=['park_geometry']),
                geometry=exact_osm_matches['greenspace_geometry']
            )
            exact_osm_matches = exact_osm_matches.drop(columns=['greenspace_geometry'])
            exact_osm_matches['match_method'] = 'exact_name_osm'
            exact_osm_matches['match_quality'] = 1.0
            
            results.append(exact_osm_matches)
            matched_indices = exact_osm_matches['index'].tolist()
            unmatched = unmatched[~unmatched.index.isin(matched_indices)]
            print(f"Found {len(exact_osm_matches)} exact OSM name matches")
            
            for _, row in exact_osm_matches.iterrows():
                match_log.append({
                    'park_name': row['Park Name'],
                    'matched_to': row['Name (OSM)'],
                    'method': 'exact_name_osm',
                    'match_quality': 1.0
                })
    
    # Strategy 3: Fuzzy name matching
    print("\n=== Strategy 3: Fuzzy Name Matching ===")
    if len(unmatched) > 0:
        fuzzy_matches = []
        for park_idx, park in unmatched.iterrows():
            best_match = None
            best_score = 0
            best_source = None
            
            # Check OS names
            if 'Name (OS)' in greenspace_gdf.columns:
                for gs_idx, gs in greenspace_gdf.iterrows():
                    if pd.notna(gs['Name (OS)']):
                        score = fuzzy_match_score(park['Park Name'], gs['Name (OS)'])
                        if score > best_score and score >= name_threshold:
                            best_match = gs_idx
                            best_score = score
                            best_source = 'os'
            
            # Check OSM names
            if 'Name (OSM)' in greenspace_gdf.columns:
                for gs_idx, gs in greenspace_gdf.iterrows():
                    if pd.notna(gs['Name (OSM)']):
                        score = fuzzy_match_score(park['Park Name'], gs['Name (OSM)'])
                        if score > best_score and score >= name_threshold:
                            best_match = gs_idx
                            best_score = score
                            best_source = 'osm'
            
            if best_match is not None:
                matched_gs = greenspace_gdf.loc[best_match]
                
                # Create a proper merged row by combining the data
                merged_data = {}
                
                # Add park data (excluding geometry)
                for col in park.index:
                    if col != 'geometry':
                        merged_data[col] = park[col]
                
                # Add greenspace data (excluding geometry, avoid duplicates)
                for col in matched_gs.index:
                    if col not in merged_data and col != 'geometry':
                        merged_data[col] = matched_gs[col]
                
                # Add match metadata
                merged_data['match_method'] = f'fuzzy_name_{best_source}'
                merged_data['match_quality'] = best_score
                
                # Create GeoDataFrame with greenspace geometry ONLY
                merged_row = gpd.GeoDataFrame([merged_data], geometry=[matched_gs.geometry])
                fuzzy_matches.append(merged_row)
                
                matched_name = matched_gs[f'Name ({best_source.upper()})']
                match_log.append({
                    'park_name': park['Park Name'],
                    'matched_to': matched_name,
                    'method': f'fuzzy_name_{best_source}',
                    'match_quality': best_score
                })
        
        if fuzzy_matches:
            fuzzy_df = gpd.GeoDataFrame(pd.concat(fuzzy_matches, ignore_index=True))
            results.append(fuzzy_df)
            matched_indices = [park_idx for park_idx, _ in unmatched.iterrows() 
                             if any(park_name in [m['park_name'] for m in match_log[-len(fuzzy_matches):]] 
                                  for park_name in [unmatched.loc[park_idx, 'Park Name']])]
            unmatched = unmatched[~unmatched.index.isin(matched_indices)]
            print(f"Found {len(fuzzy_matches)} fuzzy name matches")
    
    # Strategy 4: Spatial matching with increasing buffer sizes
    print("\n=== Strategy 4: Spatial Matching ===")
    for buffer_size in spatial_buffers:
        if len(unmatched) == 0:
            break
            
        print(f"Trying {buffer_size}m buffer...")
        
        # Create buffered greenspace if buffer > 0
        if buffer_size > 0:
            gs_buffered = greenspace_gdf.copy()
            gs_buffered['geometry'] = greenspace_gdf.geometry.buffer(buffer_size)
        else:
            gs_buffered = greenspace_gdf
        
        # Convert unmatched parks to same CRS
        parks_for_spatial = unmatched.to_crs(gs_buffered.crs)
        
        # Spatial join
        spatial_matches = gpd.sjoin(
            parks_for_spatial, 
            gs_buffered, 
            predicate='intersects', 
            how='inner'
        )
        
        if len(spatial_matches) > 0:
            # Replace buffered geometries with original greenspace geometries
            original_geometries = greenspace_gdf.loc[spatial_matches['index_right'], 'geometry'].values
            
            # Remove the current geometry column and set the original greenspace geometry
            spatial_matches = spatial_matches.drop(columns=['geometry'])
            spatial_matches = gpd.GeoDataFrame(spatial_matches, geometry=original_geometries)
            
            # Add quality metrics
            spatial_matches['match_method'] = f'spatial_{buffer_size}m'
            # Quality decreases with buffer size
            spatial_matches['match_quality'] = max(0.1, 1.0 - (buffer_size / 200))
            
            results.append(spatial_matches)
            matched_indices = spatial_matches['index'].tolist()
            unmatched = unmatched[~unmatched.index.isin(matched_indices)]
            
            print(f"Found {len(spatial_matches)} matches with {buffer_size}m buffer")
            
            # Log spatial matches
            for _, row in spatial_matches.iterrows():
                match_log.append({
                    'park_name': row['Park Name'],
                    'matched_to': f"Spatial match ({buffer_size}m)",
                    'method': f'spatial_{buffer_size}m',
                    'match_quality': row['match_quality']
                })
    
    # Combine all results
    if results:
        all_matched = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))
        # Remove duplicates (prefer higher quality matches)
        all_matched = all_matched.sort_values('match_quality', ascending=False)
        all_matched = all_matched.drop_duplicates(subset=['index'], keep='first')
        
        # Ensure we only have one geometry column
        geometry_cols = [col for col in all_matched.columns if 'geometry' in col.lower()]
        if len(geometry_cols) > 1:
            print(f"Warning: Multiple geometry columns found: {geometry_cols}")
            # Keep only the main geometry column
            for col in geometry_cols:
                if col != 'geometry':
                    all_matched = all_matched.drop(columns=[col])
    else:
        all_matched = gpd.GeoDataFrame()
    
    # Create match quality summary
    match_quality_df = pd.DataFrame(match_log)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Successfully matched: {len(all_matched)} parks")
    print(f"Unmatched: {len(unmatched)} parks")
    
    if len(match_quality_df) > 0:
        print("\nMatch method summary:")
        print(match_quality_df['method'].value_counts())
        print(f"\nAverage match quality: {match_quality_df['match_quality'].mean():.3f}")
    
    if len(unmatched) > 0:
        print(f"\nUnmatched parks: {unmatched['Park Name'].tolist()}")
    
    return all_matched, unmatched, match_quality_df
