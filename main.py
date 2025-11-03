import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import requests
import json
from shapely.geometry import Point, LineString
from scipy import stats
import seaborn as sns
from datetime import datetime, timedelta

class EnhancedInfinitySeismicModel:
    """
    Enhanced implementation of the Infinity Seismic Model with real data integration,
    advanced statistics, and predictive capabilities.
    """
    
    def __init__(self):
        self.icosahedron_vertices = None
        self.icosahedron_edges = None
        self.proximity_corridor_km = 150
        self.real_seismic_data = None
        self.major_faults_data = None
        
    def generate_icosahedron(self):
        """Generate a spherical icosahedron with proper edge connectivity."""
        phi = (1 + np.sqrt(5)) / 2
        
        vertices = np.array([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ])
        
        vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
        
        lat = np.arcsin(vertices[:, 2]) * 180 / np.pi
        lon = np.arctan2(vertices[:, 1], vertices[:, 0]) * 180 / np.pi
        lon = np.where(lon > 180, lon - 360, lon)
        
        self.icosahedron_vertices = np.column_stack([lon, lat])
        self._generate_edges()
        
        return self.icosahedron_vertices
    
    def _generate_edges(self):
        """Generate great circle edges between icosahedron vertices."""
        if self.icosahedron_vertices is None:
            self.generate_icosahedron()
            
        self.icosahedron_edges = []
        
        # Define icosahedron face connectivity
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        
        # Create edges from faces
        edge_set = set()
        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1)%3]]))
                edge_set.add(edge)
        
        # Convert to great circle arcs
        for v1, v2 in edge_set:
            edge = self._create_great_circle_arc(
                self.icosahedron_vertices[v1],
                self.icosahedron_vertices[v2]
            )
            self.icosahedron_edges.append(edge)
    
    def _spherical_distance(self, point1, point2):
        """Calculate spherical distance between two points in degrees."""
        lon1, lat1 = np.radians(point1[0]), np.radians(point1[1])
        lon2, lat2 = np.radians(point2[0]), np.radians(point2[1])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return np.degrees(c)
    
    def _create_great_circle_arc(self, start, end, num_points=100):
        """Create a great circle arc between two points."""
        start_rad = np.radians(start[::-1])
        end_rad = np.radians(end[::-1])
        
        def to_cartesian(lat, lon):
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            return np.array([x, y, z])

        def to_spherical(xyz):
            r = np.linalg.norm(xyz, axis=1)
            lat = np.arcsin(xyz[:, 2] / r)
            lon = np.arctan2(xyz[:, 1], xyz[:, 0])
            return np.degrees(lon), np.degrees(lat)

        p1 = to_cartesian(start_rad[0], start_rad[1])
        p2 = to_cartesian(end_rad[0], end_rad[1])
        
        omega = np.arccos(np.dot(p1, p2))
        
        if np.isclose(omega, 0) or np.isclose(omega, np.pi):
            return np.array([start] * num_points)

        t = np.linspace(0, 1, num_points)
        
        a = np.sin((1 - t) * omega) / np.sin(omega)
        b = np.sin(t * omega) / np.sin(omega)
        
        points_xyz = a[:, np.newaxis] * p1 + b[:, np.newaxis] * p2
        
        lons, lats = to_spherical(points_xyz)
        
        return np.column_stack([lons, lats])
    
    def fetch_real_seismic_data(self, start_date=None, end_date=None, min_magnitude=5.0):
        """Fetch real seismic data from USGS API."""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
               f"format=geojson&starttime={start_date}&endtime={end_date}"
               f"&minmagnitude={min_magnitude}")
        
        try:
            response = requests.get(url)
            data = response.json()
            
            earthquakes = []
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                earthquakes.append({
                    'magnitude': props['mag'],
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'depth': coords[2],
                    'place': props['place'],
                    'time': props['time'],
                    'type': props['type']
                })
            
            self.real_seismic_data = pd.DataFrame(earthquakes)
            print(f"Fetched {len(self.real_seismic_data)} real earthquakes")
            return self.real_seismic_data
            
        except Exception as e:
            print(f"Error fetching real data: {e}")
            return self._create_simulated_intraplate_data()
    
    def calculate_comprehensive_statistics(self, seismic_data):
        """Calculate comprehensive alignment statistics with confidence intervals."""
        if self.icosahedron_edges is None:
            self.generate_icosahedron()
        
        aligned_count = 0
        alignment_distances = []
        tolerance_deg = self.proximity_corridor_km / 111.0
        
        for _, quake in seismic_data.iterrows():
            quake_point = np.array([quake['longitude'], quake['latitude']])
            min_dist = float('inf')
            
            for edge in self.icosahedron_edges:
                for arc_point in edge[::10]:  # Sample every 10th point for efficiency
                    dist = self._spherical_distance(quake_point, arc_point)
                    if dist < min_dist:
                        min_dist = dist
            
            alignment_distances.append(min_dist * 111)  # Convert to km
            if min_dist <= tolerance_deg:
                aligned_count += 1
        
        # Statistical analysis
        total_points = len(seismic_data)
        alignment_percentage = (aligned_count / total_points) * 100
        
        # Calculate 95% confidence interval
        p = aligned_count / total_points
        se = np.sqrt(p * (1 - p) / total_points)
        ci_lower = (p - 1.96 * se) * 100
        ci_upper = (p + 1.96 * se) * 100
        
        return {
            'alignment_percentage': alignment_percentage,
            'aligned_count': aligned_count,
            'total_count': total_points,
            'confidence_interval': (ci_lower, ci_upper),
            'mean_distance_km': np.mean(alignment_distances),
            'distance_std': np.std(alignment_distances)
        }
    
    def plot_global_tectonic_map(self, seismic_data=None):
        """Create a comprehensive global tectonic map."""
        if self.icosahedron_edges is None:
            self.generate_icosahedron()
            
        fig = plt.figure(figsize=(20, 10))
        projection = ccrs.Robinson()
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines()
        
        # Plot geometric framework
        for edge in self.icosahedron_edges:
            ax.plot(edge[:, 0], edge[:, 1], color='blue', linewidth=2, 
                   transform=ccrs.PlateCarree(), alpha=0.6, label='Icosahedral Edges')
        
        ax.scatter(self.icosahedron_vertices[:, 0], self.icosahedron_vertices[:, 1], 
                  color='red', s=100, transform=ccrs.PlateCarree(), 
                  label='Vertices (Collision Zones)', zorder=5)
        
        # Plot seismic data if available
        if seismic_data is not None:
            sc = ax.scatter(seismic_data['longitude'], seismic_data['latitude'],
                          c=seismic_data['magnitude'], cmap='Reds', s=30,
                          transform=ccrs.PlateCarree(), label='Earthquakes')
            plt.colorbar(sc, ax=ax, label='Magnitude')
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.values(), loc='lower left')
        
        plt.title('Infinity Seismic Model: Global Tectonic Architecture\n' +
                 'Geometric Control of Seismic Activity', fontsize=16, fontweight='bold')
        plt.show()
    
    def predictive_hazard_analysis(self, region_bounds, resolution=1.0):
        """Generate a seismic hazard prediction map based on geometric proximity."""
        if self.icosahedron_edges is None:
            self.generate_icosahedron()
        
        lons = np.arange(region_bounds[0], region_bounds[1], resolution)
        lats = np.arange(region_bounds[2], region_bounds[3], resolution)
        
        hazard_grid = np.zeros((len(lats), len(lons)))
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                point = np.array([lon, lat])
                min_dist_km = float('inf')
                
                # Find minimum distance to any geometric edge
                for edge in self.icosahedron_edges:
                    for arc_point in edge[::20]:  # Sample for efficiency
                        dist = self._spherical_distance(point, arc_point) * 111
                        if dist < min_dist_km:
                            min_dist_km = dist
                
                # Hazard score: inverse of distance with exponential decay
                hazard_grid[i, j] = np.exp(-min_dist_km / 200)
        
        return lons, lats, hazard_grid

# Enhanced demonstration with real data analysis
def comprehensive_demonstration():
    """Run a comprehensive demonstration of the enhanced model."""
    print("🚀 ENHANCED INFINITY SEISMIC MODEL DEMONSTRATION")
    print("=" * 60)
    
    model = EnhancedInfinitySeismicModel()
    
    # 1. Generate geometric framework
    print("1. Generating icosahedral-dodecahedral lattice...")
    model.generate_icosahedron()
    print(f"   Generated {len(model.icosahedron_vertices)} vertices and {len(model.icosahedron_edges)} edges")
    
    # 2. Fetch and analyze real data
    print("2. Fetching real seismic data from USGS...")
    seismic_data = model.fetch_real_seismic_data(min_magnitude=5.5)
    
    # 3. Comprehensive statistical analysis
    print("3. Performing comprehensive statistical analysis...")
    stats = model.calculate_comprehensive_statistics(seismic_data)
    
    print("\n📊 STATISTICAL RESULTS:")
    print(f"   Total earthquakes analyzed: {stats['total_count']}")
    print(f"   Alignment percentage: {stats['alignment_percentage']:.1f}%")
    print(f"   95% Confidence Interval: [{stats['confidence_interval'][0]:.1f}%, {stats['confidence_interval'][1]:.1f}%]")
    print(f"   Mean distance to nearest edge: {stats['mean_distance_km']:.1f} km")
    
    # 4. Global visualization
    print("4. Generating global tectonic map...")
    model.plot_global_tectonic_map(seismic_data)
    
    # 5. Predictive hazard analysis for specific region
    print("5. Performing predictive hazard analysis for North America...")
    na_bounds = [-140, -60, 20, 60]
    lons, lats, hazard = model.predictive_hazard_analysis(na_bounds)
    
    # Plot hazard map
    fig, ax = plt.subplots(figsize=(12, 8))
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    ax.set_extent(na_bounds, crs=projection)
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Plot hazard contours
    contour = ax.contourf(lons, lats, hazard, levels=20, cmap='Reds', 
                         transform=projection, alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Relative Seismic Hazard Score')
    
    # Plot geometric edges
    for edge in model.icosahedron_edges:
        edge_in_region = [pt for pt in edge if (na_bounds[0] <= pt[0] <= na_bounds[1] and 
                                              na_bounds[2] <= pt[1] <= na_bounds[3])]
        if len(edge_in_region) > 5:
            edge_arr = np.array(edge_in_region)
            ax.plot(edge_arr[:, 0], edge_arr[:, 1], 'b-', linewidth=2, 
                   transform=projection, alpha=0.8)
    
    plt.title('Predictive Seismic Hazard Map\nBased on Geometric Proximity to Icosahedral Edges',
              fontsize=14, fontweight='bold')
    plt.show()
    
    print("\n✅ Demonstration completed successfully!")
    return model

if __name__ == "__main__":
    model = comprehensive_demonstration()
