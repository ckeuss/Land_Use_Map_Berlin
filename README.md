# Land Use Map Berlin 2024
## Streamlit App

### Project Overview
This interactive land use map of Berlin visualizes the proportions of different land use categories within 1000 m² tiles based on open geospatial data. Each land use polygon is intersected with a grid of square tiles, and the proportion of each land use area within each tile is calculated. These proportions are used as features for unsupervised k-means clustering, grouping tiles into five land use clusters (Recreation/ Nature/ Lakeside, Forest, Urban Residential, Agriculture, Urban-Industrial). The results are rendered on an interactive map using Folium, where each tile is color-coded and includes a popup with the land use composition. Users can input any Berlin address to locate it on the map.

### Prerequisites
Ensure that the following prerequisites are met to run the scripts in this repository:

- Python 3.x
- Streamlit (version 1.43.2)
- Required Python libraries: pandas, geopandas, numpy, shapely, scikit-learn, folium, geopy, streamlit, streamlit-folium

### Data
The data used in this Streamlit App is openly accessible [here](https://daten.berlin.de/datensaetze/alkis-berlin-tatsachliche-nutzung-wfs-0ee77a1d). It was published in 2024 by Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen Berlin.

### Access to the Streamlit App

https://landuse-berlin2024.streamlit.app/

### Feedback

If you encounter any issues, mistakes, or have suggestions for improvement, please let me know.
