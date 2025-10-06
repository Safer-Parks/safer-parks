"""
Entrance calculations for bounded areas

Algorithmic method to calculate entrances to parks based on intersecting nodes.
"""

def extract_park_entrances(park_polygon, buffer_distance=0.001):
    """
    Extract entrance points for a park polygon using network intersection analysis.
    
    Parameters:
    - park_polygon: Shapely polygon representing the park boundary
    - buffer_distance: Buffer distance for network extraction (default: 0.001 degrees)
    
    Returns:
    - entrance_points: List of raw entrance points (duplicated)
    - merged_points: Deduplicated entrance points as MultiPoint geometry
    - G: OSMnx graph for visualization
    """
    
    # Create buffered polygon for network extraction
    network_buffer = park_polygon.buffer(buffer_distance)
    
    try:
        # Extract walking network around the park
        # G = ox.graph_from_polygon(network_buffer, network_type='walk') # we may want to update this to also include bike paths
        G = ox.graph_from_polygon(network_buffer, network_type='all')
    except Exception as e:
        print(f"Error extracting network: {e}")
        return [], None, None
    
    # Find entrance points where edges cross the park boundary
    entrance_points = []
    for u, v in G.edges():
        u_pt = Point(G.nodes[u]['x'], G.nodes[u]['y'])
        v_pt = Point(G.nodes[v]['x'], G.nodes[v]['y'])
        
        # Check if edge crosses park boundary
        if park_polygon.contains(u_pt) != park_polygon.contains(v_pt):
            # Edge crosses boundary, interpolate intersection
            line = LineString([u_pt, v_pt])
            intersection = line.intersection(park_polygon.boundary)
            
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    entrance_points.append(intersection)
                elif intersection.geom_type == 'MultiPoint':
                    entrance_points.extend(intersection.geoms)
    
    # Deduplicate entrance points
    if entrance_points:
        merged_points = unary_union(entrance_points)
    else:
        merged_points = None
    
    return entrance_points, merged_points, G