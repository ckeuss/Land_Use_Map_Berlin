# Land Use Map Berlin 2024
## Streamlit App

### Project Overview
This interactive land use map of Berlin visualizes the proportions of different land use categories within 1000×1000 m² tiles based on official open geospatial data. Each land use polygon is intersected with a grid of square tiles, and the area of each land use type within each tile is calculated to ensure proportional accuracy. These area-weighted proportions are used as features for unsupervised k-means clustering, grouping tiles into five land use clusters (e.g., Urban Residential, Forest, Mixed Urban/ Recreation/ Industrial, Agriculture, Lakeside/ Nature/ Residential). The results are rendered on an interactive map using Folium, where each tile is color-coded and includes a tooltip with the land use composition. Users can input any Berlin address to locate it on the map.

### Prerequisites
Ensure that the following prerequisites are met to run the scripts in this repository:

- Python 3.x
- Streamlit (version 1.43.2)
- Required Python libraries: pandas, geopandas, numpy, shapely, scikit-learn, folium, geopy, streamlit, streamlit-folium

### Data
The data used in this Streamlit App is openly accessible [here](https://daten.berlin.de/datensaetze/alkis-berlin-tatsachliche-nutzung-wfs-0ee77a1d). It was published in 2024 by Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen Berlin.

### Access to the Streamlit App



### Feedback

If you encounter any issues, mistakes, or have suggestions for improvement, please let me know.
