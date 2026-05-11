import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from pathlib import Path
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import scipy.cluster.hierarchy as sch
from datetime import datetime
import warnings
import json

# Requisitos (instalar antes com pip):
# pip install gradio geopandas shapely scikit-learn scipy pyarrow matplotlib requests

warnings.filterwarnings("ignore")

# ==========================================
# 1. ESTRUTURA DE DIRETÓRIOS / PATHS
# ==========================================
BASE = Path(r"C:\Users\HOME\Desktop\RePotEx")
DATA_LAKE = BASE / "data_lake_inmet"

INPUT_PATH   = DATA_LAKE / "station_geo.parquet"
CLIMATE_PATH = DATA_LAKE / "station_climate.parquet"
MONTHLY_PATH = DATA_LAKE / "station_climate_monthly.parquet"
LINHAS_TRANSMISSAO_PATH = DATA_LAKE / "linhas_transmissao.parquet"

GEOJSON_DIR = BASE / "docs" / "geojson_outputs"
GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

# Tenta carregar a logo de dois caminhos possíveis
LOGO_PATH = BASE / "logoGT.png"
if not LOGO_PATH.exists():
    LOGO_PATH = DATA_LAKE / "logoGT.png"

# ==========================================
# 2. VARIÁVEIS GLOBAIS DE ESTADO E DADOS
# ==========================================
DEMOGRAPHICS = {
    'AC': {'capital': 'Rio Branco', 'pop': '829 mil', 'area': '164k km²'},
    'AL': {'capital': 'Maceió', 'pop': '3.1 milhões', 'area': '27k km²'},
    'AP': {'capital': 'Macapá', 'pop': '733 mil', 'area': '142k km²'},
    'AM': {'capital': 'Manaus', 'pop': '3.9 milhões', 'area': '1.5M km²'},
    'BA': {'capital': 'Salvador', 'pop': '14.1 milhões', 'area': '564k km²'},
    'CE': {'capital': 'Fortaleza', 'pop': '8.8 milhões', 'area': '148k km²'},
    'DF': {'capital': 'Brasília', 'pop': '2.8 milhões', 'area': '5.7k km²'},
    'ES': {'capital': 'Vitória', 'pop': '3.8 milhões', 'area': '46k km²'},
    'GO': {'capital': 'Goiânia', 'pop': '7.1 milhões', 'area': '340k km²'},
    'MA': {'capital': 'São Luís', 'pop': '6.8 milhões', 'area': '329k km²'},
    'MT': {'capital': 'Cuiabá', 'pop': '3.7 milhões', 'area': '903k km²'},
    'MS': {'capital': 'Campo Grande', 'pop': '2.8 milhões', 'area': '357k km²'},
    'MG': {'capital': 'Belo Horizonte', 'pop': '20.5 milhões', 'area': '586k km²'},
    'PA': {'capital': 'Belém', 'pop': '8.1 milhões', 'area': '1.2M km²'},
    'PB': {'capital': 'João Pessoa', 'pop': '4.0 milhões', 'area': '56k km²'},
    'PR': {'capital': 'Curitiba', 'pop': '11.4 milhões', 'area': '199k km²'},
    'PE': {'capital': 'Recife', 'pop': '9.1 milhões', 'area': '98k km²'},
    'PI': {'capital': 'Teresina', 'pop': '3.3 milhões', 'area': '251k km²'},
    'RJ': {'capital': 'Rio de Janeiro', 'pop': '16.1 milhões', 'area': '43k km²'},
    'RN': {'capital': 'Natal', 'pop': '3.3 milhões', 'area': '52k km²'},
    'RS': {'capital': 'Porto Alegre', 'pop': '10.9 milhões', 'area': '281k km²'},
    'RO': {'capital': 'Porto Velho', 'pop': '1.6 milhões', 'area': '237k km²'},
    'RR': {'capital': 'Boa Vista', 'pop': '636 mil', 'area': '224k km²'},
    'SC': {'capital': 'Florianópolis', 'pop': '7.6 milhões', 'area': '95k km²'},
    'SP': {'capital': 'São Paulo', 'pop': '44.4 milhões', 'area': '248k km²'},
    'SE': {'capital': 'Aracaju', 'pop': '2.2 milhões', 'area': '21k km²'},
    'TO': {'capital': 'Palmas', 'pop': '1.5 milhões', 'area': '277k km²'}
}

last_df_temp = None
last_cells = None
last_centroids = None
last_actual_k = 0
X_scaled = None
df_full = None
gdf_lt = None
has_lt = False
brasil_poly = None
gdf_states = None
df_monthly = None
AVAILABLE_YEARS = []

# ==========================================
# 3. CARREGAMENTO DE ATIVOS GEOGRÁFICOS
# ==========================================
def load_geographic_assets():
    global brasil_poly, gdf_states
    print("Tentando carregar ativos geográficos interativos...")
    try:
        # GeoJSON de estados do Brasil (mais detalhado para interatividade)
        url_states = "https://raw.githubusercontent.com/codeforamerica/click_container/master/Data/brazil_geo.json"
        gdf_states = gpd.read_file(url_states)
        
        # Mapeamento de propriedades para interatividade
        # brazil_geo.json costuma usar 'id' como sigla (AC, AL, etc.)
        if 'id' in gdf_states.columns:
            gdf_states['uf'] = gdf_states['id']
            gdf_states['nome_estado'] = gdf_states['uf'].map(lambda x: x) # Fallback
            gdf_states['capital'] = gdf_states['uf'].apply(lambda x: DEMOGRAPHICS.get(x, {}).get('capital', 'N/A'))
            gdf_states['populacao'] = gdf_states['uf'].apply(lambda x: DEMOGRAPHICS.get(x, {}).get('pop', 'N/A'))
            gdf_states['area_km2'] = gdf_states['uf'].apply(lambda x: DEMOGRAPHICS.get(x, {}).get('area', 'N/A'))
            
        brasil_poly = gdf_states.unary_union
        print("Estados do Brasil carregados com dados demográficos.")
        
    except Exception as e:
        print(f"Erro ao carregar ativos geográficos (Plotly): {e}")
        # Bounding box aproximada do Brasil como fallback robusto
        brasil_poly = Polygon([(-74, -34), (-34, -34), (-34, 6), (-74, 6)])
        gdf_states = gpd.GeoDataFrame({
            'uf': ['BR'], 'capital': ['Brasília'], 'populacao': ['214M'], 'geometry': [brasil_poly]
        }, crs="EPSG:4326")
        print("Usando bounding box de fallback.")

def load_data():
    global df_full, X_scaled, gdf_lt, has_lt, df_monthly, AVAILABLE_YEARS
    print(f"Iniciando carregamento de dados em: {DATA_LAKE}")
    
    # 1. Estações e Clima
    try:
        if not INPUT_PATH.exists(): raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_PATH}")
        if not CLIMATE_PATH.exists(): raise FileNotFoundError(f"Arquivo não encontrado: {CLIMATE_PATH}")
        
        df_geo = pd.read_parquet(INPUT_PATH)
        df_climate = pd.read_parquet(CLIMATE_PATH)
        print(f"Arquivos lidos: Geo({len(df_geo)}), Climate({len(df_climate)})")
        
        # Join (ajuste as chaves se necessário)
        join_key = 'station_id' if 'station_id' in df_geo.columns else df_geo.columns[0]
        df_full = pd.merge(df_geo, df_climate, on=join_key, how='inner')
        
        # Filtro geográfico básico (Brasil)
        df_full = df_full[(df_full['longitude'] > -75) & (df_full['longitude'] < -30) &
                          (df_full['latitude'] > -35) & (df_full['latitude'] < 7)]
        
        # KNN Imputer para dados climáticos
        climate_cols = df_climate.columns.drop(join_key).tolist()
        imputer = KNNImputer(n_neighbors=5)
        df_full[climate_cols] = imputer.fit_transform(df_full[climate_cols])
        
        # Scaling para Clustering (Lat/Lon)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_full[['latitude', 'longitude']])
        
    except Exception as e:
        print(f"Erro ao carregar dados principais: {e}")
        df_full = pd.DataFrame()

    # 2. Linhas de Transmissão
    try:
        if LINHAS_TRANSMISSAO_PATH.exists():
            gdf_lt = gpd.read_parquet(LINHAS_TRANSMISSAO_PATH)
            if gdf_lt.crs is None:
                gdf_lt.set_crs("EPSG:4326", inplace=True)
            elif gdf_lt.crs != "EPSG:4326":
                gdf_lt = gdf_lt.to_crs("EPSG:4326")
            
            # Limpeza e filtro
            gdf_lt = gdf_lt[gdf_lt.geometry.is_valid]
            gdf_lt = gdf_lt.cx[-75:-30, -35:7]
            has_lt = not gdf_lt.empty
        else:
            has_lt = False
    except Exception as e:
        print(f"Erro ao carregar linhas de transmissão: {e}")
        has_lt = False

    # 3. Dados Mensais
    try:
        if MONTHLY_PATH.exists():
            df_monthly = pd.read_parquet(MONTHLY_PATH)
            if 'year' in df_monthly.columns:
                AVAILABLE_YEARS = sorted(df_monthly['year'].unique().tolist())
        else:
            df_monthly = None
    except Exception as e:
        print(f"Erro ao carregar dados mensais: {e}")
        df_monthly = None

# ==========================================
# 4. LÓGICA DE VORONOI E CLUSTERING
# ==========================================
def voronoi_polygons(vor, base_shape):
    """Gera polígonos Voronoi recortados por uma forma base."""
    new_regions = []
    vol_polygons = []
    
    # Centroids
    centroids = vor.points
    
    # Reconstrução das regiões finitas/infinitas
    for i, reg_idx in enumerate(vor.point_region):
        region = vor.regions[reg_idx]
        if -1 not in region:
            # Região finita
            polygon = Polygon([vor.vertices[i] for i in region])
        else:
            # Região infinita: criar um bounding box grande
            # Simplificação: usaremos um polígono muito grande e recortaremos
            polygon = Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])
            
        # Recorte pelo Brasil
        clipped = polygon.intersection(base_shape)
        if not clipped.is_empty:
            vol_polygons.append(clipped)
        else:
            vol_polygons.append(None)
            
    return vol_polygons

def plot_linhas_transmissao(ax, label_legenda=True):
    if has_lt and gdf_lt is not None:
        gdf_lt.plot(ax=ax, color='red', linewidth=0.5, alpha=0.6, zorder=3)
        if label_legenda:
            return Line2D([0], [0], color='red', lw=1, label='Linhas de Transmissão')
    return None

def gerar_voronoi(n_raw, show_lt=True):
    global last_df_temp, last_cells, last_centroids, last_actual_k
    
    if df_full.empty:
        return None, "Dados não carregados.", None, None
    
    # Cálculo de K
    n_total = len(df_full)
    k = max(2, n_total // n_raw)
    
    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_full['cluster'] = kmeans.fit_predict(X_scaled)
    last_actual_k = k
    
    # Voronoi sobre os centróides
    centroids_scaled = kmeans.cluster_centers_
    scaler = StandardScaler().fit(df_full[['latitude', 'longitude']])
    centroids = scaler.inverse_transform(centroids_scaled)
    
    vor = Voronoi(centroids[:, [1, 0]]) 
    cells = voronoi_polygons(vor, brasil_poly)
    
    last_df_temp = df_full.copy()
    last_cells = cells
    last_centroids = centroids

    # --- CRIAÇÃO DO MAPA INTERATIVO (PLOTLY) ---
    try:
        # 1. Base de Estados com Hover Info
        # Garantindo que o GeoJSON tenha IDs consistentes
        geojson_dict = json.loads(gdf_states.to_json())
        for i, feature in enumerate(geojson_dict['features']):
            feature['id'] = str(i)

        fig = go.Figure()

        # Camada de Estados (Choropleth para Interatividade/Hover)
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson_dict,
            locations=[str(i) for i in gdf_states.index],
            z=[0] * len(gdf_states), # Cor neutra de fundo
            colorscale=[[0, 'rgba(240, 240, 240, 0.5)'], [1, 'rgba(240, 240, 240, 0.5)']],
            showscale=False,
            marker_opacity=0.4,
            marker_line_width=1,
            marker_line_color="black",
            hoverinfo="text",
            hovertext=[
                f"Estado: {r.get('uf', 'N/A')}<br>Capital: {r.get('capital', 'N/A')}<br>População: {r.get('populacao', 'N/A')}<br>Área: {r.get('area_km2', 'N/A')}"
                for _, r in gdf_states.iterrows()
            ],
            name="Brasil"
        ))

        # 2. Adicionar Células Voronoi (Coloridas por Cluster)
        cmap = plt.get_cmap('viridis', k)
        for i, cell in enumerate(cells):
            if cell:
                lons, lats = cell.exterior.coords.xy
                rgb = [int(x*255) for x in cmap(i)[:3]]
                color_hex = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)'
                fig.add_trace(go.Scattermapbox(
                    lon=list(lons),
                    lat=list(lats),
                    mode='lines',
                    line=dict(width=0.5, color='white'),
                    fill='toself',
                    fillcolor=color_hex,
                    name=f'Cluster {i}',
                    hoverinfo='name'
                ))

        # 3. Adicionar Estações
        fig.add_trace(go.Scattermapbox(
            lon=df_full['longitude'],
            lat=df_full['latitude'],
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.8),
            name='Estações',
            hovertext=df_full['station_name']
        ))

        # 4. Adicionar Linhas de Transmissão
        if show_lt and has_lt and gdf_lt is not None:
            for geom in gdf_lt.geometry:
                if geom.geom_type == 'LineString':
                    lons, lats = geom.coords.xy
                    fig.add_trace(go.Scattermapbox(
                        lon=list(lons), lat=list(lats),
                        mode='lines', line=dict(width=1.2, color='red'),
                        name='Linha de Transmissão', opacity=0.6,
                        hoverinfo='none'
                    ))

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=3.5,
            mapbox_center={"lat": -15, "lon": -55},
            margin={"r":0,"t":0,"l":0,"b":0},
            showlegend=False
        )
    except Exception as e:
        print(f"Erro ao gerar mapa interativo: {e}")
        # Fallback para Matplotlib se o Plotly falhar
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf_states.plot(ax=ax, color='white', edgecolor='black', linewidth=0.8)
        cmap = plt.get_cmap('viridis', k)
        for i, cell in enumerate(cells):
            if cell: gpd.GeoSeries(cell).plot(ax=ax, color=cmap(i), alpha=0.4, edgecolor='white', linewidth=0.5)
        ax.scatter(df_full['longitude'], df_full['latitude'], c='blue', s=2, alpha=0.5)
        if show_lt: plot_linhas_transmissao(ax)
        ax.axis('off')
        return fig, f"Erro no Plotly: {e}. Exibindo versão estática.", geo_df, clim_df
    
    summary = f"Total Estações: {n_total}\nClusters: {k}\nMédia Estações/Cluster: {n_total/k:.1f}"
    
    uf_col = 'uf' if 'uf' in df_full.columns else ('state' if 'state' in df_full.columns else df_full.columns[2])
    geo_df = df_full[[ 'station_id', 'station_name', uf_col, 'latitude', 'longitude', 'cluster']].head(50)
    climate_cols = [c for c in df_full.columns if c not in ['station_id', 'station_name', 'uf', 'state', 'latitude', 'longitude', 'cluster', 'geometry']]
    clim_df = df_full[['station_id', 'cluster'] + climate_cols[:5]].head(50)
    
    return fig, summary, geo_df, clim_df

# ==========================================
# 5. DETALHES DO CLUSTER
# ==========================================
def cluster_station_details(cluster_id):
    if last_df_temp is None: return "Gere o clustering primeiro.", None
    try:
        cid = int(cluster_id)
        df_c = last_df_temp[last_df_temp['cluster'] == cid]
        uf_col = 'uf' if 'uf' in df_c.columns else ('state' if 'state' in df_c.columns else df_c.columns[2])
        return df_c[['station_id', 'station_name', uf_col, 'latitude', 'longitude']], df_c[['latitude', 'longitude']].values.tolist()
    except:
        return "ID Inválido", None

def climate_summary_for_cluster(cluster_id):
    if last_df_temp is None: return "Gere o clustering primeiro."
    try:
        cid = int(cluster_id)
        df_c = last_df_temp[last_df_temp['cluster'] == cid]
        summary = df_c.describe().transpose()
        return summary
    except:
        return "ID Inválido"

def plot_cluster_voronoi_only(cluster_id, show_lt=True):
    if last_df_temp is None: return None
    try:
        cid = int(cluster_id)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Base Brasil cinza claro
        gdf_states.plot(ax=ax, color='#f0f0f0', edgecolor='gray', linewidth=0.5, alpha=0.3)
        
        # Célula em destaque
        if last_cells and cid < len(last_cells) and last_cells[cid]:
            gpd.GeoSeries(last_cells[cid]).plot(ax=ax, color='orange', alpha=0.5, edgecolor='darkorange', linewidth=2)
            
            # Zoom na célula
            bounds = last_cells[cid].bounds
            ax.set_xlim(bounds[0]-1, bounds[2]+1)
            ax.set_ylim(bounds[1]-1, bounds[3]+1)
        
        # Estações do cluster
        df_c = last_df_temp[last_df_temp['cluster'] == cid]
        ax.scatter(df_c['longitude'], df_c['latitude'], c='blue', s=20, label='Estações do Cluster')
        
        if show_lt: plot_linhas_transmissao(ax, label_legenda=False)
        
        ax.set_title(f"Detalhe do Cluster {cid}", fontsize=12)
        ax.legend()
        return fig
    except:
        return None

# ==========================================
# 6. BOXPLOTS
# ==========================================
def gerar_boxplot_dispatch(cluster_id, climate_var, mode, selected_stations):
    if last_df_temp is None: return None
    try:
        cid = int(cluster_id)
        df_all = last_df_temp[last_df_temp['cluster'] == cid]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if mode == "cluster":
            df_all.boxplot(column=climate_var, ax=ax, patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', color='blue'),
                          medianprops=dict(color='red'))
            ax.set_title(f"Distribuição de {climate_var} - Cluster {cid}")
            
        elif mode == "single":
            if not selected_stations: return None
            df_sub = df_all[df_all['station_id'].isin(selected_stations)]
            df_sub.boxplot(column=climate_var, by='station_id', ax=ax, patch_artist=True)
            plt.suptitle("") # Remove o título automático do pandas
            ax.set_title(f"Comparação de {climate_var} - Estações Selecionadas")
            plt.xticks(rotation=45)
            
        elif mode == "compare":
            # Similar ao single mas com todas do cluster comparadas
            df_all.boxplot(column=climate_var, by='station_id', ax=ax, patch_artist=True, widths=0.7)
            plt.suptitle("")
            ax.set_title(f"Comparação de todas as estações do Cluster {cid}")
            plt.xticks(rotation=90, fontsize=6)
            ax.margins(x=0.02)
            
        return fig
    except Exception as e:
        print(e)
        return None

def gerar_boxplot_mensal(cluster_id, climate_var, selected_years, selected_months, selected_stations, ts_mode):
    if df_monthly is None: return None, "Dados mensais não disponíveis."
    try:
        cid = int(cluster_id)
        station_ids = last_df_temp[last_df_temp['cluster'] == cid]['station_id'].tolist()
        
        df_sub = df_monthly[df_monthly['station_id'].isin(station_ids)]
        if not df_sub.empty:
            if selected_years: df_sub = df_sub[df_sub['year'].isin(selected_years)]
            if selected_months: df_sub = df_sub[df_sub['month'].isin(selected_months)]
            if selected_stations: df_sub = df_sub[df_sub['station_id'].isin(selected_stations)]
            
        if df_sub.empty: return None, "Nenhum dado encontrado para os filtros selecionados."
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if ts_mode == "all_months":
            df_sub.boxplot(column=climate_var, by='month', ax=ax, patch_artist=True)
            ax.set_title(f"Variação Mensal de {climate_var} - Cluster {cid}")
        elif ts_mode == "by_year":
            df_sub.boxplot(column=climate_var, by=['year', 'month'], ax=ax, patch_artist=True)
            plt.xticks(rotation=90, fontsize=6)
            ax.set_title(f"Variação por Ano/Mês - Cluster {cid}")
        elif ts_mode == "by_station":
            df_sub.boxplot(column=climate_var, by='station_id', ax=ax, patch_artist=True)
            plt.xticks(rotation=90, fontsize=6)
            ax.set_title(f"Comparação por Estação - Cluster {cid}")
            
        plt.suptitle("")
        return fig, "Gráfico gerado com sucesso."
    except Exception as e:
        return None, f"Erro: {str(e)}"

# ==========================================
# 7. DENDROGRAMA
# ==========================================
def gerar_dendrograma(max_samples=200, method="ward"):
    if X_scaled is None: return None
    try:
        # Subamostragem para performance
        indices = np.random.choice(len(X_scaled), min(len(X_scaled), max_samples), replace=False)
        X_sub = X_scaled[indices]
        
        linked = sch.linkage(X_sub, method=method)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sch.dendrogram(linked, ax=ax, orientation='top', distance_sort='descending', show_leaf_counts=True)
        ax.set_title(f"Agrupamento Hierárquico (Dendrograma) - {method}")
        return fig
    except:
        return None

# ==========================================
# 8. INTERFACE GRADIO
# ==========================================

CSS = """
.gradio-container { background-color: #f7f9fc; font-family: 'Inter', sans-serif; }
.header { text-align: center; padding: 20px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; border-radius: 10px; margin-bottom: 20px; }
.section-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px; }
h1, h2 { margin: 0; }
"""

def create_ui():
    load_geographic_assets()
    load_data()
    
    with gr.Blocks(css=CSS, title="RePoTEx") as app:
        # Header
        with gr.Row(elem_classes=["header"]):
            with gr.Column(scale=1):
                if LOGO_PATH.exists():
                    gr.Image(str(LOGO_PATH), show_label=False, height=80, width=80)
            with gr.Column(scale=4):
                gr.HTML("<h1>RePoTEx</h1><p>Sistema de Análise Climática e Regionalização</p>")
        
        # Status LT
        lt_status = "✅ Linhas de Transmissão Carregadas" if has_lt else "⚠️ Linhas de Transmissão NÃO Disponíveis"
        gr.HTML(f"<div style='padding: 5px; text-align: center; background: #eef2ff; border-radius: 5px;'>{lt_status}</div>")
        
        # --- SEÇÃO 1: CLUSTERING ---
        with gr.Tab("Clustering & Voronoi"):
            with gr.Row():
                with gr.Column(scale=1):
                    n_raw = gr.Slider(minimum=5, maximum=100, value=30, step=5, label="Estações por Cluster (Aprox.)")
                    show_lt_check = gr.Checkbox(value=True, label="Mostrar Linhas de Transmissão")
                    btn_run = gr.Button("Gerar Diagrama de Voronoi", variant="primary")
                    summary_box = gr.Textbox(label="Resumo do Clustering", lines=3)
                with gr.Column(scale=3):
                    plot_voronoi = gr.Plot(label="Mapa de Voronoi - Brasil")
            
            with gr.Row():
                geo_table = gr.DataFrame(label="Dados Geográficos (Top 50)")
                clim_table = gr.DataFrame(label="Dados Climáticos (Top 50)")

        # --- SEÇÃO 2: DETALHES DO CLUSTER ---
        with gr.Tab("Detalhes por Cluster"):
            with gr.Row():
                with gr.Column(scale=1):
                    cluster_id_input = gr.Number(value=0, label="ID do Cluster")
                    btn_detail = gr.Button("Ver Detalhes")
                    cluster_summary = gr.DataFrame(label="Estatísticas do Cluster")
                with gr.Column(scale=2):
                    plot_cluster = gr.Plot(label="Mapa Detalhado do Cluster")
                    station_list = gr.DataFrame(label="Estações no Cluster")

        # --- SEÇÃO 3: BOXPLOTS CLIMÁTICOS ---
        with gr.Tab("Boxplots Climáticos"):
            with gr.Row():
                with gr.Column(scale=1):
                    bp_cluster_id = gr.Number(value=0, label="Cluster ID")
                    # Ajustando escolhas para colunas reais
                    climate_choices = ["temp_mean", "temp_max", "temp_min", "precip_mean", "precip_sum", "wind_mean", "rh_mean"]
                    bp_var = gr.Dropdown(choices=climate_choices, value="temp_mean", label="Variável Climática")
                    bp_mode = gr.Radio(choices=["cluster", "single", "compare"], value="cluster", label="Modo")
                    bp_stations = gr.Dropdown(choices=[], multiselect=True, label="Estações (Para modo Single/Compare)")
                    btn_boxplot = gr.Button("Gerar Boxplot")
                with gr.Column(scale=2):
                    plot_boxplot = gr.Plot()

        # --- SEÇÃO 4: SÉRIES TEMPORAIS MENSAL ---
        with gr.Tab("Análise Mensal (Time-Series)"):
            with gr.Row():
                with gr.Column(scale=1):
                    ts_cluster_id = gr.Number(value=0, label="Cluster ID")
                    # Ajustando escolhas para colunas reais
                    ts_choices = ["temp_mean", "precip_mean", "rh_mean"]
                    ts_var = gr.Dropdown(choices=ts_choices, value="temp_mean", label="Variável")
                    ts_years = gr.Dropdown(choices=AVAILABLE_YEARS, multiselect=True, label="Anos")
                    ts_mode = gr.Radio(choices=["all_months", "by_year", "by_station"], value="all_months", label="Visualização")
                    btn_ts = gr.Button("Gerar Análise Mensal")
                with gr.Column(scale=2):
                    plot_ts = gr.Plot()
                    ts_status = gr.Markdown()

        # --- SEÇÃO 5: DENDROGRAMA ---
        with gr.Tab("Dendrograma"):
            with gr.Row():
                with gr.Column(scale=1):
                    max_s = gr.Slider(50, 500, 200, step=50, label="Amostras Máximas")
                    method = gr.Dropdown(["ward", "single", "complete", "average"], value="ward", label="Método")
                    btn_dendro = gr.Button("Gerar Dendrograma")
                with gr.Column(scale=2):
                    plot_dendro = gr.Plot()

        # CALLBACKS
        btn_run.click(gerar_voronoi, inputs=[n_raw, show_lt_check], outputs=[plot_voronoi, summary_box, geo_table, clim_table])
        
        def update_station_dropdown(cluster_id):
            if last_df_temp is not None:
                stations = last_df_temp[last_df_temp['cluster'] == int(cluster_id)]['station_id'].unique().tolist()
                return gr.Dropdown(choices=stations)
            return gr.Dropdown(choices=[])

        btn_detail.click(cluster_station_details, inputs=[cluster_id_input], outputs=[station_list, gr.State()])
        btn_detail.click(climate_summary_for_cluster, inputs=[cluster_id_input], outputs=[cluster_summary])
        btn_detail.click(plot_cluster_voronoi_only, inputs=[cluster_id_input, show_lt_check], outputs=[plot_cluster])
        btn_detail.click(update_station_dropdown, inputs=[cluster_id_input], outputs=[bp_stations])

        btn_boxplot.click(gerar_boxplot_dispatch, inputs=[bp_cluster_id, bp_var, bp_mode, bp_stations], outputs=[plot_boxplot])
        btn_ts.click(gerar_boxplot_mensal, inputs=[ts_cluster_id, ts_var, ts_years, gr.State([]), gr.State([]), ts_mode], outputs=[plot_ts, ts_status])
        btn_dendro.click(gerar_dendrograma, inputs=[max_s, method], outputs=[plot_dendro])

    return app

if __name__ == "__main__":
    dashboard = create_ui()
    dashboard.launch(share=False, debug=False)
