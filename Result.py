
"""
Aplicação: identificação de hotspots e modelo preditivo de acidentes na base PRF.
Saída:
 - indicadores resumo (totais e percentuais)
 - GeoJSONs: hotspots_h3.geojson, kde_hotspots.geojson, clusters.geojson
 - mapa: hotspots_prf.html
 - métricas de avaliação do modelo (PR-AUC, HitRate@100m)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import h3
import json
import folium
from datetime import timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from scipy.spatial import cKDTree
from math import radians, cos, sin, asin, sqrt

# ========== CONFIG ==========
DATA_PATH = "prf_acidentes_U8-CV.csv"   # altere para o caminho do CSV baixado da PRF
OUTPUT_DIR = "output_prf"
H3_RES = 8                              # resolução H3 (ajuste conforme necessidade)
PRED_HORIZON_DAYS = 4                   # horizonte para rótulo futuro
MIN_DATE_FOR_MODEL = None               # se quiser limitar janelas temporais
SAMPLE_MARKERS = 500                    # número de pontos a plotar como marcadores (amostra)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ========== UTILS ==========
def haversine(lon1, lat1, lon2, lat2):
    # distancia em metros entre dois pontos
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c

def ensure_datetime(df, col):
    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    return df

# ========== 1) LEITURA E NORMALIZAÇÃO DE COLUNAS ==========
print("1) Carregando dados...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Mapear nomes de coluna comuns (adapte se necessário)
COL_MAP = {
    # tenta encontrar colunas com os nomes entre colchetes
    #    "date": ["data_inversa", "data", "datahora", "data_hora", "date"],
    #    "causa": ["causa_acidente", "causa", "cause"],
    #    "gravidade": ["classificacao_acidente", "gravidade", "severity", "classificação"],
    #    "latitude": ["latitude", "lat", "latitudine"],
    #    "longitude": ["longitude", "lon", "lng", "long"]
    #   'id':['id'],
    'date':['date'],
    'causa':['causa_acidente'],
    'gravidade':['gravidade'],
    'latitude':['latitude'],
    'longitude':['longitude']
}

found = {}

for canonical, candidates in COL_MAP.items():
    for c in candidates:
        print("valor do c...",c)
        if c in df.columns[0]:
            found[canonical] = c
            print("valor do [canonical]...",canonical)
            print("valor do found[canonical]...",found[canonical])
            break


# Renomear
df = df.rename(columns={'date':'date', 'latitude':'latitude', 'longitude':'longitude'})
if "gravidade" in found:
    df = df.rename(columns={found["gravidade"]:"gravidade"})
if "causa" in found:
    df = df.rename(columns={found["causa"]:"causa"})

# Limpar nulos e conversões
df = df.dropna(subset=['date', 'latitude', 'longitude'])
df["latitude"] = df["latitude"].astype(float)
df["longitude"] = df["longitude"].astype(float)
df = ensure_datetime(df, "date")
df = df.dropna(subset=["date"])

print(f"Registros carregados: {len(df)}")

# ========== 2) INDICADORES GERAIS ==========
print("2) Calculando indicadores gerais...")
total = len(df)

# Normalizar gravidade (tentativa simples)
if 'gravidade' in df.columns:
    gvals = df['gravidade'].astype(str)
    # heurística para classificar
    fatal_mask = gvals.str.contains("Fatais|óbito|óbito|morte|mortes|fatalidade")
    injured_mask = gvals.str.contains("Feridas|lesionado|injured|injury")
    minor_mask = ~(fatal_mask | injured_mask)
    
else:
    # se não houver coluna, estimar tudo como leve
    fatal_mask = pd.Series(False, index=df.index)

    injured_mask = pd.Series(False, index=df.index)
    minor_mask = pd.Series(True, index=df.index)

n_fatal = fatal_mask.sum()
n_injured = injured_mask.sum()
n_minor = minor_mask.sum()

pct_fatal = 100 * n_fatal / total if total else 0
pct_injured = 100 * n_injured / total if total else 0
pct_minor = 100 * n_minor / total if total else 0

print(f"Totais -> Acidentes: {total}, Leves: {n_minor} ({pct_minor:.1f}%), Feridos: {n_injured} ({pct_injured:.1f}%), Fatais: {n_fatal} ({pct_fatal:.1f}%)")

# ========== 3) AGREGAÇÃO DE H3 ==========
print("3) Agregando por H3...")
df["h3"] = df.apply(lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_RES), axis=1)
h3_counts = df.groupby("h3").size().reset_index(name="n_acidentes")
# percentual do total concentrado no topo da célula
h3_counts = h3_counts.sort_values("n_acidentes", ascending=False)
top5pct_cells = int(max(1, 0.05 * len(h3_counts)))
top_cells = h3_counts.head(top5pct_cells)
prop_top5pct = 100 * top_cells["n_acidentes"].sum() / total   
print(f"{len(h3_counts)} células H3. Top 5% células concentram {prop_top5pct:.1f}% dos acidentes")

# Código para versão mais recente de h3

def h3_to_polygon(h):
    # Retorna lista de (lat, lon)
    coords = h3.cell_to_boundary(h)
    # Inverte para (lon, lat) pois o GeoJSON espera essa ordem
    coords = [(lon, lat) for lat, lon in coords]
    return Polygon(coords)

# conversão de colunas com códigos H3 para polígonos
h3_counts["geometry"] = h3_counts["h3"].apply(h3_to_polygon)

# Cria GeoDataFrame com CRS correto
gdf_h3 = gpd.GeoDataFrame(h3_counts, geometry="geometry", crs="EPSG:4326")

# Exporta para GeoJSON
OUTPUT_DIR = "saida"
os.makedirs(OUTPUT_DIR, exist_ok=True)
gdf_h3.to_file(os.path.join(OUTPUT_DIR, "hotspots_h3.geojson"), driver="GeoJSON")


# ========== 4) KDE (opcional) - criar raster/grid de densidade e extrair hotspots ==========
print("4) Gerando hotspots por KDE (grade)...")
# criação de grade simples (em mercator) para KDE
gpts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326").to_crs(epsg=3857)
minx, miny, maxx, maxy = gpts.total_bounds
cell = 500  # tamanho célula em metros para KDE grid
xs = np.arange(minx, maxx+1, cell)
ys = np.arange(miny, maxy+1, cell)
xx, yy = np.meshgrid(xs, ys)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
# converter coordenadas de incidentes para marcador
coords = np.vstack([gpts.geometry.x.values, gpts.geometry.y.values]).T
# O KDE simples via scipy gaussian_kde tem alto custo; aproximado agrupando as contagens de bin na grade
xi = np.digitize(gpts.geometry.x.values, xs) - 1
yi = np.digitize(gpts.geometry.y.values, ys) - 1
bins = {}
for i,j in zip(xi, yi):
    if 0 <= i < len(xs)-1 and 0 <= j < len(ys)-1:
        bins[(i,j)] = bins.get((i,j),0)+1
# converte bins para geo dataframe
polys = []
vals = []
for (i,j), cnt in bins.items():
    x0 = xs[i]; y0 = ys[j]
    poly = Polygon([(x0,y0),(x0+cell,y0),(x0+cell,y0+cell),(x0,y0+cell)])
    polys.append(poly)
    vals.append(cnt)
gdf_kde = gpd.GeoDataFrame({"cnt": vals}, geometry=polys, crs="EPSG:3857").to_crs(epsg=4326)
# limite superior de 5% das células de densidade
thr = np.quantile(gdf_kde["cnt"], 0.95)
gdf_kde_hot = gdf_kde[gdf_kde["cnt"] >= thr]
gdf_kde_hot.to_file(os.path.join(OUTPUT_DIR, "kde_hotspots.geojson"), driver="GeoJSON")
print(f"KDE hotspots: {len(gdf_kde_hot)} células (threshold quantil 95%)")

# ========== 5) CRIAÇÃO DE FEATURES TEMPORAIS E LABELS (PREDIÇÃO POR CÉLULA H3) ==========
print("5) Construindo features e labels para modelagem...")

# preparar series diárias por célula
df["date_day"] = df["date"].dt.floor("D")
pv = df.groupby(["date_day","h3"]).size().rename("cnt").reset_index()
# pivot: rows = date, cols = h3 ids
pivot = pv.pivot_table(index="date_day", columns="h3", values="cnt", fill_value=0).sort_index()
# limita datas se for rquerido
if MIN_DATE_FOR_MODEL:
    pivot = pivot[pivot.index >= pd.to_datetime(MIN_DATE_FOR_MODEL)]

# criar recursos de atraso por célula
def build_feature_matrix(pivot, lags=[1,7,14]):
    # pivot: DataFrame indexado por data, colunas por h3
    frames = []
    for lag in lags:
        frames.append(pivot.shift(lag).add_suffix(f"_lag{lag}"))
    Xwide = pd.concat(frames, axis=1).dropna(how='all')  # drop top datas com NaNs para todas as celulas
    # pilha muito longa: cada row = (data, h3, features...)
    Xs = Xwide.stack().rename("val").reset_index()  # isso só dá um valor; em vez disso, realizando iteração
    # Empilhando com retardo
    rows = []
    dates = Xwide.index
    for date in dates:
        row = Xwide.loc[date]
        
    records = []
    for cell in pivot.columns:
        df_cell = pd.DataFrame({
            "date": pivot.index,
            "cell": cell,
            "cnt": pivot[cell].values
        })
        for lag in [1,7,14]:
            df_cell[f"lag{lag}"] = df_cell["cnt"].shift(lag)
        df_cell = df_cell.dropna(subset=["lag1"])  # ensure lags exist
        records.append(df_cell)
    data_ml = pd.concat(records, ignore_index=True)
   
    data_ml[["lag1","lag7","lag14"]] = data_ml[["lag1","lag7","lag14"]].fillna(0)
    return data_ml

data_ml = build_feature_matrix(pivot)
print("Records for ML:", len(data_ml))

# criar rótulos: uma célula teve >=1 acidente nos próximos PRED_HORIZON_DAYS?
data_ml = data_ml.sort_values(["cell","date"])
data_ml["future_cnt_sum"] = data_ml.groupby("cell")["cnt"].transform(lambda s: s.shift(-1).rolling(PRED_HORIZON_DAYS, min_periods=1).sum())
data_ml["label"] = (data_ml["future_cnt_sum"] > 0).astype(int)

# corta os últimos dias onde a janela futura está incompleta
last_valid_date = pivot.index.max() - pd.Timedelta(days=PRED_HORIZON_DAYS)
data_ml = data_ml[data_ml["date"] <= last_valid_date].copy()

# simples features e label
features = ["lag1","lag7","lag14"]
X = data_ml[features]
y = data_ml["label"].values

# ========== 6) TREINO/AVALIAÇÃO TEMPORAL ==========
print("6) Treinando e avaliando modelo (TimeSeriesSplit)...")
ts = TimeSeriesSplit(n_splits=6)
aps = []
models = []
for fold, (train_idx, test_idx) in enumerate(ts.split(X)):
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    # usar LightGBM se instalado
    try:
        dtrain = lgb.Dataset(Xtr, label=ytr)
        params = {"objective":"binary", "metric":"auc", "verbosity":-1, "boosting_type":"gbdt"}
        bst = lgb.train(params, dtrain, num_boost_round=200)
        proba = bst.predict(Xte)
        models.append(bst)
    except Exception:
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:,1]
        models.append(clf)
    aps.append(average_precision_score(yte, proba))
    print(f"Fold {fold+1} AP: {aps[-1]:.3f}")

print(f"PR-AUC médio: {np.mean(aps):.3f}")

# traino final em todos os dados
print("Treinando modelo final em todos os dados disponíveis...")
try:
    dtrain = lgb.Dataset(X, label=y)
    final = lgb.train({"objective":"binary","metric":"auc","verbosity":-1}, dtrain, num_boost_round=200)
    def predict_fun(Xp): return final.predict(Xp)
except Exception:
    final_clf = GradientBoostingClassifier(random_state=42)
    final_clf.fit(X, y)
    def predict_fun(Xp): return final_clf.predict_proba(Xp)[:,1]

# ========== 7) PREVISÕES RECENTES E HIT RATE@100m ==========
print("7) Gerando previsões e avaliando HitRate@100m...")
# calcular as previsões de data mais recentes (para o dia = last_valid_date)
pred_date = last_valid_date
latest_rows = data_ml[data_ml["date"]==pred_date].copy()
latest_rows["score"] = predict_fun(latest_rows[features])
# top K cells (K = 100)
K = 100
topk = latest_rows.sort_values("score", ascending=False).head(K)
topk_cells = topk["cell"].unique()

# preparar pontos de acidentes reais futuros nos próximos PRED_HORIZON_DAYS
start = pred_date + pd.Timedelta(days=1)
end = pred_date + pd.Timedelta(days=PRED_HORIZON_DAYS)
future_pts = df[(df["date_day"]>=start) & (df["date_day"]<=end)].copy()
print(f"Acidentes no horizonte futuro ({start.date()} a {end.date()}): {len(future_pts)}")

# calcula os centroides do topk h3 celulas
topk_coords = []
for c in topk_cells:
    lat, lon = h3.cell_to_latlng(c)  # returns (lat, lon) tuple
    topk_coords.append((lon, lat))  # note ordering
    
    

# constroe kdtree para vizinho mais próximo rápido (usar lon/lat aproximado; para maior precisão transformar para metros)
if len(topk_coords) == 0 or len(future_pts) == 0:
    hitrate = 0.0
else:
    # para cada ponto futuro, verifica se está dentro de 100 m de qualquer centróide topk
    hits = 0
    for _, r in future_pts.iterrows():
        lonp, latp = r["longitude"], r["latitude"]
        near = False
        for lonc, latc in topk_coords:
            d = haversine(lonc, latc, lonp, latp)
            if d <= 100.0:
                near = True
                break
        if near:
            hits += 1
    hitrate = 100.0 * hits / len(future_pts)
print(f"HitRate@100m (Top {K} cells): {hitrate:.1f}%")

# ========== 8) EXPORTS: GeoJSON + Folium map ==========
print("8) Exportando GeoJSON e mapa interativo...")
# Exporta topk h3 polygons
topk_polys = [h3.cell_to_boundary(c) for c in topk_cells]
polys = [Polygon(coords) for coords in topk_polys]
gdf_topk = gpd.GeoDataFrame({"h3":list(topk_cells)}, geometry=polys, crs="EPSG:4326")
gdf_topk.to_file(os.path.join(OUTPUT_DIR, "topk_hotspots.geojson"), driver="GeoJSON")

# Folium map
m = folium.Map(location=[df["latitude"].median(), df["longitude"].median()], zoom_start=5, tiles="CartoDB positron")
# adiciona KDE hotspots
folium.GeoJson(gdf_kde_hot.to_json(), name="KDE Hotspots", style_function=lambda feat: {"fillColor":"red","color":"red","weight":1,"fillOpacity":0.4}).add_to(m)
# adiciona topk cells
folium.GeoJson(gdf_topk.to_json(), name="Top{K} Predicted", style_function=lambda feat: {"fillColor":"blue","color":"blue","weight":1,"fillOpacity":0.3}).add_to(m)
# adiciona pontos amostrados
for _, r in df.sample(min(SAMPLE_MARKERS, len(df))).iterrows():
    folium.CircleMarker(location=[r["latitude"], r["longitude"]], radius=2, color="black", fill=True, fillOpacity=0.6).add_to(m)
m.save(os.path.join(OUTPUT_DIR, "hotspots_prf.html"))

# ========== 9) Relatório resumo (imprime comparações com valores reais apresentados no relatório) ==========
print("\n=== RELATÓRIO RESUMO ===")
print(f"Total acidentes (base): {total}")
print(f"Leves: {n_minor} ({pct_minor:.1f}%) | Feridos: {n_injured} ({pct_injured:.1f}%) | Fatais: {n_fatal} ({pct_fatal:.1f}%)")
print(f"Top 5% células concentram {prop_top5pct:.1f}% dos acidentes")
print(f"PR-AUC médio (validação temporal): {np.mean(aps):.3f}")
print(f"HitRate@100m (Top {K}): {hitrate:.1f}%")
print(f"Arquivos gerados em: {OUTPUT_DIR} -> hotspots_h3.geojson, kde_hotspots.geojson, topk_hotspots.geojson, hotspots_prf.html")

# salvar indicadores em CSV
pd.DataFrame([{
    "total": total,
    "n_minor": n_minor, "pct_minor": pct_minor,
    "n_injured": n_injured, "pct_injured": pct_injured,
    "n_fatal": n_fatal, "pct_fatal": pct_fatal,
    "top5pct_prop": prop_top5pct,
    "pr_auc": np.mean(aps),
    "hitrate_topk_100m": hitrate
}]).to_csv(os.path.join(OUTPUT_DIR, "indicadores_resumo.csv"), index=False)

print("Pronto.")
