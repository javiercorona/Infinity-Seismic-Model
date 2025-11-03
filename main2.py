# corrected_enhanced_ism.py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import requests
from shapely.geometry import Point, LineString
from scipy import stats
from datetime import datetime, timedelta

class EnhancedInfinitySeismicModel:
    """
    Enhanced implementation with robustness fixes, a simulated-data fallback,
    and vectorized distance evaluation.
    """
    def __init__(self):
        self.icosahedron_vertices = None
        self.icosahedron_edges = None
        self.edge_sample_points = None  # pre-sampled points (lon, lat) along all edges
        self.proximity_corridor_km = 150
        self.real_seismic_data = None
        self.major_faults_data = None

    # ---------- Geometry ----------
    def generate_icosahedron(self):
        """Generate a spherical icosahedron with proper edge connectivity."""
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ])
        vertices = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]
        lat = np.degrees(np.arcsin(vertices[:, 2]))
        lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
        lon = np.where(lon > 180, lon - 360, lon)
        self.icosahedron_vertices = np.column_stack([lon, lat])
        self._generate_edges()
        self._pre_sample_edges(sample_per_edge=100)  # precompute for vectorization
        return self.icosahedron_vertices

    def _generate_edges(self):
        """Generate great circle edges between icosahedron vertices."""
        if self.icosahedron_vertices is None:
            self.generate_icosahedron()
        self.icosahedron_edges = []
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        edge_set = set()
        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_set.add(edge)
        for v1, v2 in edge_set:
            edge = self._create_great_circle_arc(
                self.icosahedron_vertices[v1], self.icosahedron_vertices[v2], num_points=200
            )
            self.icosahedron_edges.append(edge)

    def _create_great_circle_arc(self, start, end, num_points=100):
        """
        Create a great circle arc between two points in degrees.
        start/end are [lon, lat] in degrees.
        """
        # Convert to radians and to Cartesian vectors
        start_rad = np.radians(start[::-1])  # -> [lat, lon]
        end_rad = np.radians(end[::-1])
        def to_cartesian(lat, lon):
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            return np.array([x, y, z])
        p1 = to_cartesian(start_rad[0], start_rad[1])
        p2 = to_cartesian(end_rad[0], end_rad[1])
        dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
        omega = np.arccos(dot)
        if np.isclose(omega, 0) or np.isclose(omega, np.pi):
            return np.tile(np.array(start), (num_points, 1))
        t = np.linspace(0, 1, num_points)
        a = np.sin((1 - t) * omega) / np.sin(omega)
        b = np.sin(t * omega) / np.sin(omega)
        points_xyz = a[:, None] * p1 + b[:, None] * p2
        # normalize
        points_xyz /= np.linalg.norm(points_xyz, axis=1)[:, None]
        lon = np.degrees(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]))
        lat = np.degrees(np.arcsin(points_xyz[:, 2]))
        lon = np.where(lon > 180, lon - 360, lon)
        return np.column_stack([lon, lat])

    def _pre_sample_edges(self, sample_per_edge=100):
        """Create a single array of sampled (lon, lat) points for all edges for vectorized distance calcs."""
        pts = []
        for edge in self.icosahedron_edges:
            # sample uniformly along arc; edge already has many points so take a subset:
            idx = np.linspace(0, len(edge) - 1, sample_per_edge).astype(int)
            sampled = edge[idx]
            pts.append(sampled)
        if pts:
            self.edge_sample_points = np.vstack(pts)
        else:
            self.edge_sample_points = np.empty((0, 2))

    # ---------- Utilities (Vectorized) ----------
    @staticmethod
    def _haversine_deg(lon1, lat1, lon2_arr, lat2_arr):
        """
        Vectorized haversine distance in degrees between a single lon1,lat1 and arrays lon2_arr,lat2_arr.
        Returns central angle in degrees.
        """
        # convert to radians
        lon1r, lat1r = np.radians(lon1), np.radians(lat1)
        lon2r = np.radians(lon2_arr)
        lat2r = np.radians(lat2_arr)
        dlon = lon2r - lon1r
        dlat = lat2r - lat1r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        # avoid slight numerical overflow
        a = np.clip(a, 0.0, 1.0)
        c = 2 * np.arcsin(np.sqrt(a))
        return np.degrees(c)

    # ---------- Data fetch ----------
    def fetch_real_seismic_data(self, start_date=None, end_date=None, min_magnitude=5.0, limit=20000):
        """Fetch real seismic data from USGS API. Converts times to datetime and handles HTTP errors."""
        if start_date is None:
            start_date = (datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        url = (
            "https://earthquake.usgs.gov/fdsnws/event/1/query?"
            f"format=geojson&starttime={start_date}&endtime={end_date}"
            f"&minmagnitude={min_magnitude}&limit={limit}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            earthquakes = []
            for feature in data.get('features', []):
                props = feature['properties']
                geom = feature.get('geometry', {})
                coords = geom.get('coordinates', [None, None, None])
                # convert epoch ms to datetime
                time_ms = props.get('time', None)
                dt = None
                if time_ms is not None:
                    dt = datetime.utcfromtimestamp(int(time_ms) / 1000.0)
                earthquakes.append({
                    'magnitude': props.get('mag'),
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'depth': coords[2] if len(coords) > 2 else None,
                    'place': props.get('place'),
                    'time': dt,
                    'type': props.get('type')
                })
            self.real_seismic_data = pd.DataFrame(earthquakes).dropna(subset=['longitude', 'latitude'])
            print(f"Fetched {len(self.real_seismic_data)} real earthquakes")
            return self.real_seismic_data
        except Exception as e:
            print(f"Error fetching real data: {e}")
            return self._create_simulated_intraplate_data(n=500)

    def _create_simulated_intraplate_data(self, n=300):
        """Create a simple simulated catalog of intraplate quakes (fallback)."""
        rng = np.random.default_rng(42)
        lons = rng.uniform(-180, 180, n)
        lats = rng.uniform(-60, 75, n)
        mags = rng.uniform(5.0, 7.5, n)
        df = pd.DataFrame({'longitude': lons, 'latitude': lats, 'magnitude': mags})
        self.real_seismic_data = df
        print(f"Created {len(df)} simulated earthquakes")
        return df

    # ---------- Analysis (Vectorized) ----------
    def calculate_comprehensive_statistics(self, seismic_data):
        """Calculate comprehensive alignment statistics with confidence intervals (vectorized)."""
        if self.icosahedron_edges is None or self.edge_sample_points is None:
            self.generate_icosahedron()

        # tolerance in degrees (approx 1 deg ~ 111 km)
        tolerance_deg = self.proximity_corridor_km / 111.0

        all_min_dists_km = []

        # prepare arrays for vectorized haversine
        edge_lons = self.edge_sample_points[:, 0]
        edge_lats = self.edge_sample_points[:, 1]

        # Use iterrows for structure, but the core distance calculation is vectorized
        for _, quake in seismic_data.iterrows():
            qlon, qlat = quake['longitude'], quake['latitude']
            # vectorized computation against all sampled edge points
            central_angle_deg = self._haversine_deg(qlon, qlat, edge_lons, edge_lats)
            min_deg = np.min(central_angle_deg)
            min_km = min_deg * 111.0
            all_min_dists_km.append(min_km)

        aligned_mask = np.array(all_min_dists_km) <= self.proximity_corridor_km
        aligned_count = int(np.sum(aligned_mask))
        total_points = len(seismic_data)
        alignment_percentage = (aligned_count / total_points) * 100 if total_points > 0 else 0.0

        # Agresti-Coull 95% CI (better for proportions)
        if total_points > 0:
            n = total_points
            x = aligned_count
            z = 1.96
            n_hat = n + z**2
            p_hat = (x + 0.5 * z**2) / n_hat
            se = np.sqrt(p_hat * (1 - p_hat) / n_hat)
            ci_lower = max(0.0, (p_hat - z * se)) * 100
            ci_upper = min(1.0, (p_hat + z * se)) * 100
        else:
            ci_lower, ci_upper = 0.0, 0.0

        return {
            'alignment_percentage': alignment_percentage,
            'aligned_count': aligned_count,
            'total_count': total_points,
            'confidence_interval': (ci_lower, ci_upper),
            'mean_distance_km': float(np.mean(all_min_dists_km)) if all_min_dists_km else None,
            'distance_std': float(np.std(all_min_dists_km)) if all_min_dists_km else None
        }

    # ---------- Visualization ----------
    def plot_global_tectonic_map(self, seismic_data=None):
        """Create a comprehensive global tectonic map."""
        if self.icosahedron_edges is None:
            self.generate_icosahedron()
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines()
        # Plot edges
        for edge in self.icosahedron_edges:
            ax.plot(edge[:, 0], edge[:, 1], linewidth=2, transform=ccrs.PlateCarree(),
                    alpha=0.6, label='Icosahedral Edges')
        # Vertices
        ax.scatter(self.icosahedron_vertices[:, 0], self.icosahedron_vertices[:, 1],
                   color='red', s=100, transform=ccrs.PlateCarree(),
                   label='Vertices (Collision Zones)', zorder=5)
        if seismic_data is not None:
            sc = ax.scatter(seismic_data['longitude'], seismic_data['latitude'],
                            c=seismic_data.get('magnitude', 0), cmap='Reds', s=30,
                            transform=ccrs.PlateCarree(), label='Earthquakes')
            plt.colorbar(sc, ax=ax, label='Magnitude')
        # Fix legend: extract unique handles/labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower left')
        plt.title('Infinity Seismic Model: Global Tectonic Architecture\n'
                  'Geometric Control of Seismic Activity', fontsize=16, fontweight='bold')
        plt.show()

    # ---------- Predictive hazard (Vectorized) ----------
    def predictive_hazard_analysis(self, region_bounds, resolution=1.0):
        """Generate a seismic hazard prediction map based on geometric proximity (vectorized)."""
        if self.icosahedron_edges is None or self.edge_sample_points is None:
            self.generate_icosahedron()
        lon_min, lon_max, lat_min, lat_max = region_bounds
        lons = np.arange(lon_min, lon_max + 1e-6, resolution)
        lats = np.arange(lat_min, lat_max + 1e-6, resolution)
        hazard_grid = np.zeros((len(lats), len(lons)))
        edge_lons = self.edge_sample_points[:, 0]
        edge_lats = self.edge_sample_points[:, 1]
        for i, lat in enumerate(lats):
            # vectorize across longitudes by broadcasting
            for j, lon in enumerate(lons):
                central_deg = self._haversine_deg(lon, lat, edge_lons, edge_lats)
                min_dist_km = np.min(central_deg) * 111.0
                hazard_grid[i, j] = np.exp(-min_dist_km / 200.0)
        return lons, lats, hazard_grid

# Enhanced demonstration with real data analysis
def comprehensive_demonstration():
    print("🚀 ENHANCED INFINITY SEISMIC MODEL DEMONSTRATION")
    print("=" * 60)
    model = EnhancedInfinitySeismicModel()
    print("1. Generating icosahedral-dodecahedral lattice...")
    model.generate_icosahedron()
    print(f"   Generated {len(model.icosahedron_vertices)} vertices and {len(model.icosahedron_edges)} edges")
    print("2. Fetching real seismic data from USGS...")
    seismic_data = model.fetch_real_seismic_data(min_magnitude=5.5)
    print("3. Performing comprehensive statistical analysis...")
    stats = model.calculate_comprehensive_statistics(seismic_data)
    print("\n📊 STATISTICAL RESULTS:")
    print(f"   Total earthquakes analyzed: {stats['total_count']}")
    print(f"   Alignment percentage: {stats['alignment_percentage']:.1f}%")
    print(f"   95% Confidence Interval: [{stats['confidence_interval'][0]:.1f}%, {stats['confidence_interval'][1]:.1f}%]")
    print(f"   Mean distance to nearest edge: {stats['mean_distance_km']:.1f} km")
    print("4. Generating global tectonic map...")
    model.plot_global_tectonic_map(seismic_data)
    print("5. Performing predictive hazard analysis for North America...")
    na_bounds = [-140, -60, 20, 60]
    lons, lats, hazard = model.predictive_hazard_analysis(na_bounds, resolution=1.0)
    # Quick hazard plot
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(na_bounds, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    contour = ax.contourf(lons, lats, hazard, levels=20, cmap='Reds', transform=ccrs.PlateCarree(), alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Relative Seismic Hazard Score')
    # Plot edges clipped to region
    for edge in model.icosahedron_edges:
        arr = np.array(edge)
        mask = (arr[:, 0] >= na_bounds[0]) & (arr[:, 0] <= na_bounds[1]) & (arr[:, 1] >= na_bounds[2]) & (arr[:, 1] <= na_bounds[3])
        if np.count_nonzero(mask) > 2:
            ax.plot(arr[mask, 0], arr[mask, 1], 'b-', linewidth=2, transform=ccrs.PlateCarree(), alpha=0.8)
    plt.title('Predictive Seismic Hazard Map\nBased on Geometric Proximity to Icosahedral Edges', fontsize=14, fontweight='bold')
    plt.show()
    print("\n✅ Demonstration completed successfully!")
    return model

if __name__ == "__main__":
    model = comprehensive_demonstration()
