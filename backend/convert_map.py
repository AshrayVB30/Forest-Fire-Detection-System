import geopandas as gpd

# Load shapefile (your path)
shapefile_path = r"E:\IPL\pythonProject\forest-fire-detection\data\Maps\Admin2.shp"

# Read shapefile
gdf = gpd.read_file(shapefile_path)

# Save to GeoJSON
geojson_path = r"E:\IPL\pythonProject\forest-fire-detection\data\Maps\india_states.geojson"
gdf.to_file(geojson_path, driver="GeoJSON")

print("✅ GeoJSON saved at:", geojson_path)
