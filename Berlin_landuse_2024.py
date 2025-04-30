# Libraries

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import folium
from geopy.geocoders import Nominatim
import streamlit as st
from streamlit_folium import st_folium

# Streamlit APP

# Streamlit set up
st.set_page_config(page_title="Land Use Map Berlin", layout = "wide")
st.title("Interactive Land Use Map Berlin 2024")


col1, col2 = st.columns([1.5,0.5])


with col2:

    # Text and form
    st.markdown(
    """
    This land use map of Berlin shows the proportions of land use categories within 1000×1000 m² map tiles. Hover over the tiles to see the tooltips.
    The five colors represent land use patterns identified through k-means clustering, an unsupervised machine learning method. 
    Cluster labels were assigned based on the average composition of land use types within each group. The geospatial data from 2024 is sourced from [Berlin Open Data Portal](https://daten.berlin.de/datensaetze/alkis-berlin-tatsachliche-nutzung-wfs-0ee77a1d).<br>
    - <span style='color:#F43C3D'><b>Urban Residential</b></span>  
    - <span style='color:#33BBC1'><b>Forest</b></span>  
    - <span style='color:#F9DF6A'><b>Mixed Urban/ Recreation/ Industrial</b></span>  
    - <span style='color:#D5E1E4'><b>Agriculture</b></span>  
    - <span style='color:#34ADE2'><b>Lakeside/ Nature/ Residential</b></span>  
    """,
    unsafe_allow_html=True
)

    with st.form("address_form", clear_on_submit=True):
        st.write("Map Location")
        st.session_state["address"] = st.text_input("Insert an address to show the location on the map (e.g. Brandenburger Tor 1, Berlin):")
        submit = st.form_submit_button("Update Map")


# Load data
#WFS: https://fbinter.stadt-berlin.de/fb/wfs/data/senstadt/s_wfs_alkis_tatsaechlichenutzungflaechen
gdf = gpd.read_file("data/berlin_landuse.geojson")

# Drop columns with all NaN values
gdf.dropna(axis=1, how="all", inplace=True)

# Keep only bezeich and bezfkt 
columns = ["nam", "fkt", "bezfkt", "zus", "bezzus", "hyd", "bezhyd", "uuid"]
gdf.drop(columns = columns, inplace=True)

# Get rid of prefix AX_
gdf["bezeich"] = gdf["bezeich"].str.replace("^AX_", "", regex=True)



#####land use data in a grid of square tiles#####

# Tile size (in meters)
tile_size = 1000

# Bounding box
minx, miny, maxx, maxy = gdf.total_bounds

# Grid of square tiles
tiles = []
for x in np.arange(minx, maxx, tile_size):
    for y in np.arange(miny, maxy, tile_size):
        tiles.append(box(x, y, x + tile_size, y + tile_size))

tile_gdf = gpd.GeoDataFrame(geometry=tiles, crs=gdf.crs)

# tile_id as column
tile_gdf = tile_gdf.reset_index().rename(columns={"index": "tile_id"})

# Intersect each land use polygon with each tile 
intersection = gpd.overlay(gdf, tile_gdf, how="intersection")

# Add area column
intersection["area"] = intersection.geometry.area

# Group by tile and land use type
area_by_tile = intersection.groupby(["tile_id", "bezeich"])["area"].sum().unstack(fill_value=0)

# Normalize to proportions per tile (so each tile sums to ca. 1.0)
tile_features = area_by_tile.div(area_by_tile.sum(axis=1), axis=0)

# Merge proportions back to tile_gdf for mapping
tile_gdf = tile_gdf.join(tile_features, on="tile_id")
tile_features.sum(axis=1).round(3).value_counts()

# In case of empty intersections
tile_features = tile_features.fillna(0)

##### k means clustering #####

# KMeans clusters
k = 5

kmeans = KMeans(n_clusters=k, random_state=42)
tile_features["cluster"] = kmeans.fit_predict(tile_features)

# Preserve tile_id
tile_features["tile_id"] = tile_features.index

# Merge cluster values into tile_gdf
tile_gdf = tile_gdf.merge(tile_features["cluster"], on="tile_id", how="left")

# Exclude unassigned tiles, no overlapping polygons
tile_gdf = tile_gdf[tile_gdf["cluster"].notna()]

# Data type
tile_gdf["cluster"] = tile_gdf["cluster"].astype(int).astype(str)

# Characteristics of the clusters: mean values
cluster_means = tile_gdf.drop(columns=["geometry", "tile_id"]).groupby("cluster").mean()

# Iterate over each cluster and print its means
for cluster_id in cluster_means.index:
    print(f"Means for Cluster {cluster_id}:")
    print(cluster_means.loc[cluster_id])
    print("\n")


print(tile_gdf["cluster"].isna().sum())


##### Folium Map #####

# Cluster label mapping
cluster_labels = {
    "0": "Urban Residential",
    "1": "Forest",
    "2": "Mixed Urban/ Recreation/ Industrial",
    "3": "Agriculture",
    "4": "Lakeside/ Nature/ Residential"
}

color_mapping = {
    "Urban Residential": "#F43C3D",    
    "Forest": "#33BBC1",    
    "Mixed Urban/ Recreation/ Industrial": "#F9DF6A",  
    "Agriculture": "#D5E1E4",
    "Lakeside/ Nature/ Residential": "#34ADE2"  
}

# Add cluster names
tile_gdf["cluster_label"] = tile_gdf["cluster"].map(cluster_labels)

# Add colors
tile_gdf["color"] = tile_gdf["cluster_label"].map(color_mapping)

# CRS
tile_gdf = tile_gdf.to_crs(epsg=4326)

# Session state 
if "location" not in st.session_state:
    st.session_state["location"] = None
if "address" not in st.session_state:
    st.session_state["address"] = None


with col1:
    default_location = [52.5200, 13.4050]
    map_center = default_location

    m = folium.Map(location=map_center, zoom_start=10)

    cluster_colors = {
        "0": "#F43C3D",
        "1": "#33BBC1",
        "2": "#F9DF6A",
        "3": "#D5E1E4",
        "4": "#34ADE2"
    }

    for _, row in tile_gdf.iterrows():
        landuse_columns = ['Bahnverkehr', 'FlaecheBesondererFunktionalerPraegung', 'FlaecheGemischterNutzung',
            'Fliessgewaesser', 'Flugverkehr', 'Friedhof', 'Gehoelz', 'Hafenbecken', 'Halde', 'Heide',
            'IndustrieUndGewerbeflaeche', 'Landwirtschaft', 'Moor', 'Platz', 'Schiffsverkehr',
            'SportFreizeitUndErholungsflaeche', 'StehendesGewaesser', 'Strassenverkehr', 'Sumpf',
            'TagebauGrubeSteinbruch', 'UnlandVegetationsloseFlaeche', 'Wald', 'Weg', 'Wohnbauflaeche']
        
        tooltip = "<br>".join(
            [f"{col}: {row[col]*100:.2f}%" for col in landuse_columns if col in row]
        )

        folium.GeoJson(
            row["geometry"],
            style_function=lambda feature, color=cluster_colors.get(str(row["cluster"]), "gray"): {
                "fillColor": color,
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.5,
            },
            tooltip=tooltip
        ).add_to(m)

    # If submit add geocoded location
    if submit and st.session_state["address"]:
    
        geolocator = Nominatim(user_agent="landuse_mapper")
        location = geolocator.geocode(st.session_state["address"])
        if location:
            st.session_state["location"] = [location.latitude, location.longitude]
            st.session_state["address"] = location.address
        else:
            st.warning("Address not found.")
            st.session_state["location"] = None
            st.session_state["address"] = None

    # Render marker if location is available
    if st.session_state["location"]:
        folium.Marker(
            location=st.session_state["location"],
            popup=f"Address: {st.session_state.get('address', 'Selected address')}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

    # Render map
    # suppress interaction-driven reruns
    map_output = st_folium(m, width=800, height=600, returned_objects=[])
