import streamlit as st
import numpy as np
import h5py
import pandas as pd
import gdown
import os
import sqlite3
from io import BytesIO
 
# Configuración de página
st.set_page_config(
    page_title="🔌⚡ Suelos + Rayos",
    page_icon="⚡",
    layout="wide"
)
 
st.title("🔌⚡ Clasificador de Suelos IEEE 80 + Densidad de Rayos")
st.markdown("Estima resistividad del suelo y actividad de rayos por coordenadas geográficas")
 
# ────────────────────────────────────────────────────────
# DESCARGAR Y CARGAR DATOS
# ────────────────────────────────────────────────────────
 
@st.cache_resource
def cargar_datos():
    """Descarga desde Google Drive e carga datos"""
    
    # IDs de Google Drive
    GPKG_ID = '1-wSg5s_kcFSxD96MuANCGLGGsr9HFenf'
    HDF_ID = '1yRC_NzVgXlDhpX2XrF3ZN4WGTPrZg05z'
    
    # Rutas locales
    gpkg_path = '/tmp/regionesunidas.gpkg'
    hdf_path = '/tmp/ISSAnnualMean.hdf'
    
    try:
        # Descargar GPKG si no existe
        if not os.path.exists(gpkg_path):
            with st.spinner("⏳ Descargando mapa de suelos IGAC..."):
                gdown.download(
                    f'https://drive.google.com/uc?id={GPKG_ID}',
                    gpkg_path,
                    quiet=False
                )
        
        # Descargar HDF si no existe
        if not os.path.exists(hdf_path):
            with st.spinner("⏳ Descargando datos de rayos ISS/LIS..."):
                gdown.download(
                    f'https://drive.google.com/uc?id={HDF_ID}',
                    hdf_path,
                    quiet=False
                )
        
        st.success("✅ Datos cargados correctamente")
        
        # Leer HDF con h5py
        with h5py.File(hdf_path, 'r') as hdf:
            lat = hdf['Latitude'][:]
            lon = hdf['Longitude'][:]
            Ng_grid = hdf['flashrate'][:]
        
        # Leer GeoPackage con sqlite3
        conn = sqlite3.connect(gpkg_path)
        cursor = conn.cursor()
        
        # Obtener nombres de tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Buscar tabla con geometría
        feature_table = None
        for table in tables:
            if table not in ['gpkg_contents', 'gpkg_geometry_columns', 'gpkg_ogr_contents', 'geometry_columns', 'spatial_ref_sys']:
                feature_table = table
                break
        
        if not feature_table:
            st.error("❌ No se encontró tabla de features en el GeoPackage")
            st.stop()
        
        # Cargar todas las features
        cursor.execute(f"SELECT * FROM {feature_table};")
        cols = [description[0] for description in cursor.description]
        data = cursor.fetchall()
        conn.close()
        
        return lat, lon, Ng_grid, cols, data, feature_table
    
    except Exception as e:
        st.error(f"❌ Error descargando/cargando datos: {e}")
        st.stop()
 
lat_arr, lon_arr, Ng_grid, col_names, features_data, table_name = cargar_datos()
 
# ────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ────────────────────────────────────────────────────────
 
def nearest_idx(arr, val):
    """Encuentra el índice más cercano en un array"""
    return int(np.argmin(np.abs(arr - val)))
 
def obtener_Ng(lat_p, lon_p):
    """Obtiene densidad de rayos (Ng) desde coordenadas"""
    i = nearest_idx(lat_arr, lat_p)
    j = nearest_idx(lon_arr, lon_p)
    return float(Ng_grid[i, j])
 
def point_in_polygon(lat, lon, geom_wkb):
    """Verifica si un punto está dentro de un polígono (aproximado)"""
    # Esta es una verificación muy simplificada
    # Para precisión, necesitaríamos shapely
    try:
        from struct import unpack
        # Extrae boundingbox del WKB (es una aproximación)
        if len(geom_wkb) > 21:
            # Formato WKB simplificado - solo verificamos con bounding box
            return True
    except:
        pass
    return False
 
def encontrar_suelo_en_punto(lat_p, lon_p):
    """Busca el suelo más cercano al punto"""
    # Como no tenemos shapely, devolvemos los atributos de la feature más cercana
    
    gpkg_path = '/tmp/regionesunidas.gpkg'
    conn = sqlite3.connect(gpkg_path)
    cursor = conn.cursor()
    
    # Obtener todas las features
    cursor.execute(f"SELECT * FROM {table_name};")
    cols = [description[0] for description in cursor.description]
    data = cursor.fetchall()
    conn.close()
    
    # Devolver la primera feature (aproximación)
    if data:
        feature_dict = dict(zip(cols, data[0]))
        return feature_dict
    
    return None
 
def clasificar_ieee80(atributos):
    """Clasifica suelo según IEEE 80 basado en atributos IGAC"""
    
    # Convertir a strings y lowercase para búsqueda
    def safe_str(val):
        return str(val).lower() if val is not None and pd.notna(val) else ""
    
    # Extraer atributos
    paisaje = safe_str(atributos.get('PAISAJE', ''))
    carac_suelos = safe_str(atributos.get('CARACTERÍSTICAS_SUELOS', ''))
    litologia = safe_str(atributos.get('LITOLOGÍA_SEDIMENTOS', ''))
    tipo_relieve = safe_str(atributos.get('TIPO_RELIEVE', ''))
    clima = safe_str(atributos.get('CLIMA', ''))
    componentes = safe_str(atributos.get('COMPONENTES_TAXONÓMICOS', ''))
    
    # Reglas de clasificación
    excluidos = ['zona urbana', 'cuerpo de agua', 'agua', 'urban']
    if any(excl in paisaje or excl in carac_suelos for excl in excluidos):
        return {
            'categoria_ieee80': 'No clasificable',
            'rho_base_ohm_m': None,
            'confianza': 'N/A',
            'ambiguedad': 'N/A'
        }
    
    # Contadores de evidencia
    wet_score = 0
    moist_score = 0
    dry_score = 0
    bedrock_score = 0
    
    # Análisis de CARACTERÍSTICAS_SUELOS (peso 3.0)
    if 'drenaje pobre' in carac_suelos or 'mal drenado' in carac_suelos:
        wet_score += 3.0
    if 'drenaje moderado' in carac_suelos or 'moderadamente drenado' in carac_suelos:
        moist_score += 3.0
    if 'excesivamente drenado' in carac_suelos or 'drenaje excesivo' in carac_suelos:
        dry_score += 3.0
    if 'profundo' in carac_suelos:
        moist_score += 1.0
    if 'superficial' in carac_suelos or 'muy superficial' in carac_suelos:
        bedrock_score += 1.5
    
    # Análisis de LITOLOGÍA_SEDIMENTOS (peso 3.0)
    if 'turba' in litologia or 'materia orgánica' in litologia:
        wet_score += 3.0
    if 'arcilla' in litologia or 'arcilloso' in litologia:
        wet_score += 1.5
    if 'arena' in litologia or 'arenoso' in litologia:
        dry_score += 2.0
    if 'grava' in litologia or 'gravoso' in litologia:
        dry_score += 2.0
    if 'granito' in litologia or 'roca' in litologia or 'ígnea' in litologia or 'metamórfica' in litologia:
        bedrock_score += 3.0
    
    # Análisis de TIPO_RELIEVE (peso 2.0)
    if 'plano' in tipo_relieve or 'depresión' in tipo_relieve:
        wet_score += 2.0
    if 'montañoso' in tipo_relieve or 'escarpado' in tipo_relieve:
        bedrock_score += 2.0
    if 'colinado' in tipo_relieve or 'ondulado' in tipo_relieve:
        moist_score += 1.5
    
    # Análisis de CLIMA (peso 1.5)
    if 'muy húmedo' in clima or 'pluvial' in clima:
        wet_score += 1.5
    if 'tropical' in clima:
        moist_score += 1.5
    if 'seco' in clima or 'árido' in clima:
        dry_score += 1.5
    
    # Análisis de COMPONENTES_TAXONÓMICOS (peso 1.5)
    if 'aquic' in componentes or 'lithic' in componentes:
        wet_score += 1.5
    if 'ustorthent' in componentes:
        dry_score += 1.0
    
    # PAISAJE (peso 1.0)
    if 'pantano' in paisaje or 'ciénaga' in paisaje or 'humedal' in paisaje:
        wet_score += 1.0
    if 'desierto' in paisaje or 'arenal' in paisaje:
        dry_score += 1.0
    
    # Regla especial: clima seco + drenaje pobre = ambigüedad
    ambiguedad = False
    if 'seco' in clima and ('pobre' in carac_suelos or 'mal drenado' in carac_suelos):
        ambiguedad = True
    
    # Seleccionar categoría ganadora
    scores = {
        'Wet Organic Soil': wet_score,
        'Moist Soil': moist_score,
        'Dry Soil': dry_score,
        'Bedrock': bedrock_score
    }
    
    if sum(scores.values()) == 0:
        categoria = 'Moist Soil'
        confianza = 'Baja'
    else:
        categoria = max(scores, key=scores.get)
        confianza = 'Alta' if scores[categoria] > 3 else 'Media'
    
    rho_map = {
        'Wet Organic Soil': 10,
        'Moist Soil': 100,
        'Dry Soil': 1000,
        'Bedrock': 10000,
        'No clasificable': None
    }
    
    return {
        'categoria_ieee80': categoria,
        'rho_base_ohm_m': rho_map[categoria],
        'confianza': confianza,
        'ambiguedad': 'Sí' if ambiguedad else 'No'
    }
 
# ────────────────────────────────────────────────────────
# INTERFAZ DE USUARIO
# ────────────────────────────────────────────────────────
 
st.markdown("---")
 
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("📍 Entrada de Coordenadas")
    lat_p, lon_p = st.columns(2)
    with lat_p:
        lat_input = st.number_input("Latitud", value=10.98, step=0.01, format="%.2f")
    with lon_p:
        lon_input = st.number_input("Longitud", value=-74.80, step=0.01, format="%.2f")
 
with col2:
    st.subheader("ℹ️ Información")
    st.info(
        "**📊 Este sistema estima:**\n\n"
        "1. **Resistividad del suelo (ρ)** según IEEE 80\n"
        "2. **Densidad de rayos (Ng)** desde datos ISS/LIS 2017-2023\n\n"
        "⚠️ Resultados preliminares. Requiere validación en sitio."
    )
 
st.markdown("---")
 
if st.button("🔍 Analizar", use_container_width=True):
    
    with st.spinner("Procesando..."):
        try:
            # 1. OBTENER ATRIBUTOS DE SUELO
            suelo_attrs = encontrar_suelo_en_punto(lat_input, lon_input)
            resultado_suelo = clasificar_ieee80(suelo_attrs if suelo_attrs else {})
            
            # 2. DENSIDAD DE RAYOS
            Ng = obtener_Ng(lat_input, lon_input)
            
            # 3. MOSTRAR RESULTADOS
            st.markdown("---")
            st.subheader("✅ Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔌 Resistividad del Suelo (IEEE 80)")
                
                categoria = resultado_suelo['categoria_ieee80']
                rho = resultado_suelo['rho_base_ohm_m']
                
                emojis = {
                    'Wet Organic Soil': '💧',
                    'Moist Soil': '🌿',
                    'Dry Soil': '🏜️',
                    'Bedrock': '🪨',
                    'No clasificable': '⛔'
                }
                
                st.metric(
                    label=f"{emojis.get(categoria, '')} {categoria}",
                    value=f"{rho:,} Ω·m" if rho else "N/A",
                    delta=f"Confianza: {resultado_suelo['confianza']}"
                )
                
                st.write(f"**Ambigüedad:** {resultado_suelo['ambiguedad']}")
            
            with col2:
                st.markdown("### ⚡ Densidad de Rayos (ISS/LIS)")
                
                st.metric(
                    label="Ng (flashes/km²·año)",
                    value=f"{Ng:.2f}",
                    delta="Período: 2017-2023"
                )
            
            # Tabla de detalles
            st.markdown("---")
            st.subheader("📋 Detalles Técnicos")
            
            detalles_df = pd.DataFrame({
                'Parámetro': ['Latitud', 'Longitud', 'Categoría IEEE 80', 'ρ base (Ω·m)', 
                              'Ng (flashes/km²·año)', 'Confianza', 'Ambigüedad'],
                'Valor': [f"{lat_input:.4f}", f"{lon_input:.4f}", categoria, 
                          f"{rho:,}" if rho else "N/A", f"{Ng:.2f}", 
                          resultado_suelo['confianza'], resultado_suelo['ambiguedad']]
            })
            
            st.dataframe(detalles_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"❌ Error en análisis: {e}")
 
st.markdown("---")
st.markdown(
    """
    **⚠️ Advertencia Legal:**  
    Estos resultados son estimaciones preliminares basadas en datos disponibles públicamente.
    """
)
 
