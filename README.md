# RePoTEx

**Voronoi Geographic Clustering & Climate Analysis — INMET Weather Stations**

Dashboard interativo para análise climática e clusterização geográfica de estações
meteorológicas do INMET, com suporte a linhas de transmissão elétrica e visualização
de mapas interativos.

---

## Visão Geral

O RePoTEx agrupa estações meteorológicas brasileiras por proximidade geográfica
utilizando K-Means e diagramas de Voronoi. Para cada cluster, o sistema calcula
médias climáticas, distâncias até linhas de transmissão (LT) e exibe tudo em um
dashboard Gradio com mapas estáticos (Matplotlib) e interativos (Folium/Leaflet).

---

## Funcionalidades

- Clusterização geográfica via K-Means com diagrama de Voronoi recortado no polígono do Brasil
- Mapa interativo Leaflet com tooltips por cluster e popup por estação
- Heatmap climático por variável (temperatura, precipitação, vento, umidade etc.)
- Boxplots agregados e mensais por cluster e por estação
- Cálculo de distância Haversine até a linha de transmissão mais próxima (cKDTree)
- Dendrograma de clustering hierárquico
- Exportação automática de células Voronoi em GeoJSON
- Suporte a dados mensais para análise de séries temporais

---

## Estrutura de Pastas

RePoTEx/ ├── app.py ├── logoGT.png ├── requirements.txt ├── data_lake_inmet/ │ ├── station_geo.parquet │ ├── station_climate.parquet │ ├── station_climate_monthly.parquet # opcional │ └── linhas_transmissao.parquet # opcional └── geojson_outputs/ # criado automaticamente
