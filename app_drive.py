import streamlit as st
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pyhdf.SD import SD, SDC
import pandas as pd
import gdown
import os

# Configuración de página
st.set_page_config(
    page_title="🔌⚡ Suelos + Rayos",
    page_icon="⚡",
    layout="wide"
)

st.title("🔌⚡ Clasificador de Suelos IEEE 80 + Densidad de Rayos")
st.markdown("Estima resistividad del suelo y actividad de rayos por coordenadas geográficas")

# ────────────────────────────────────────────────────────
# DESCARGAR Y CARGAR DATOS (con cache)
# ────────────────────────────────────────────────────────

@st.cache_resource
def cargar_datos():
    """Descarga desde Google Drive e carga GeoPackage y HDF"""
    
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
        
        # Cargar datos
        st.success("✅ Datos cargados correctamente")
        
        gdf = gpd.read_file(gpkg_path)
        hdf = SD(hdf_path, SDC.READ)
        lat = hdf.select("Latitude")[:]
        lon = hdf.select("Longitude")[:]
        Ng_grid = hdf.select("flashrate")[:]
        
        return gdf, lat, lon, Ng_grid
    
    except Exception as e:
        st.error(f"❌ Error descargando/cargando datos: {e}")
        st.stop()

gdf, lat_arr, lon_arr, Ng_grid = cargar_datos()

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

def clasificar_ieee80(atributos):
    """Clasifica suelo según IEEE 80 basado en atributos IGAC"""
    
    # Convertir a strings y lowercase para búsqueda
    def safe_str(val):
        return str(val).lower() if pd.notna(val) else ""
    
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
            'ambiguedad': 'N/A',
            'justificacion_breve': 'Zona urbana o cuerpo de agua'
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
        dry_score += 1.0  # pero no es determinante
    
    # PAISAJE (peso 1.0) - muy general
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
    
    # Si todas las puntuaciones son 0, por defecto Moist Soil
    if sum(scores.values()) == 0:
        categoria = 'Moist Soil'
        confianza = 'Baja'
    else:
        categoria = max(scores, key=scores.get)
        confianza = 'Alta' if scores[categoria] > 3 else 'Media'
    
    # Asignar resistividad
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
        'ambiguedad': 'Sí' if ambiguedad else 'No',
        'justificacion_breve': f"Score: Wet={wet_score:.1f}, Moist={moist_score:.1f}, Dry={dry_score:.1f}, Bedrock={bedrock_score:.1f}"
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

# Botón de análisis
if st.button("🔍 Analizar", use_container_width=True):
    
    with st.spinner("Procesando..."):
        
        # 1. CLASIFICACIÓN DE SUELOS
        punto = Point(lon_input, lat_input)
        pto_gdf = gpd.GeoDataFrame([{'id': 1}], geometry=[punto], crs='EPSG:4326')
        
        try:
            hit = gpd.sjoin(pto_gdf, gdf, how='left', predicate='intersects')
            
            if hit.empty or all(hit[col].isna().all() for col in hit.columns if col != 'geometry'):
                st.error("❌ Punto fuera de cobertura del mapa IGAC")
            else:
                suelo_attrs = hit.iloc[0].to_dict()
                resultado_suelo = clasificar_ieee80(suelo_attrs)
                
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
                
                # Información de capa IGAC (si disponible)
                st.markdown("---")
                st.subheader("🗺️ Atributos IGAC Detectados")
                
                cols_relevantes = ['PAISAJE', 'CARACTERÍSTICAS_SUELOS', 'LITOLOGÍA_SEDIMENTOS', 
                                  'TIPO_RELIEVE', 'CLIMA', 'COMPONENTES_TAXONÓMICOS']
                
                igac_info = {}
                for col in cols_relevantes:
                    if col in suelo_attrs and pd.notna(suelo_attrs[col]):
                        igac_info[col] = suelo_attrs[col]
                
                if igac_info:
                    for key, val in igac_info.items():
                        st.write(f"**{key}:** {val}")
                else:
                    st.info("No hay atributos IGAC disponibles para esta ubicación")
        
        except Exception as e:
            st.error(f"❌ Error en análisis: {e}")

st.markdown("---")
st.markdown(
    """
    **⚠️ Advertencia Legal:**  
    Estos resultados son estimaciones preliminares basadas en datos disponibles públicamente.  
    Para diseño definitivo de sistemas de puesta a tierra, se requiere:
    - Medición en sitio según **IEEE Std 81**
    - Estudio geotécnico profesional
    - Validación con normativa local (RETIE - Colombia)
    
    Desarrollado con datos IGAC y NASA ISS/LIS
    """
)
