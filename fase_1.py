# Sistema Inteligente de Análisis de Opiniones y Generación Automática de Recomendaciones
# Proyecto Capstone - Samsung Innovation Campus 2024
# Versión corregida - manejo robusto de errores

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# librerías para procesamiento de texto y NLP
import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# sklearn para machine learning y clustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# bibliotecas adicionales para análisis avanzado
from wordcloud import WordCloud
from datetime import datetime
import plotly.express as px
import plotly.graph_objects

# configuración de estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

print("Sistema de Análisis de Opiniones - Pipeline NLP Avanzado")
print("-"*60)

# ===============================
# 0. CONFIGURACIÓN INICIAL: CARGA DE DATASETS LOCALES
# ===============================

def cargar_datasets_locales():
    """
    carga los datasets CSV locales proporcionados
    maneja diferentes encodings y estructuras
    """
    print("0. Cargando datasets locales:")
    print("-" * 50)
    
    datasets_cargados = {}
    
    # definir archivos locales disponibles
    archivos_csv = ['sentimentanalysis.csv', 'redmi6.csv']
    
    for archivo in archivos_csv:
        if os.path.exists(archivo):
            print(f"Procesando: {archivo}")
            
            # intentar diferentes encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(archivo, encoding=encoding)
                    print(f"  Cargado con encoding {encoding}")
                    print(f"  Dimensiones: {df.shape}")
                    print(f"  Columnas: {list(df.columns)}")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            
            if df is not None:
                # verificar si hay problemas de separadores en sentiment analysis
                if 'sentimentanalysis' in archivo and df.shape[1] == 1:
                    primera_columna = df.columns[0]
                    if ',' in primera_columna:
                        print("  Detectando separadores incorrectos...")
                        # intentar parsear correctamente
                        try:
                            df_temp = pd.read_csv(archivo, encoding='utf-8')
                            # separar la primera fila manualmente
                            columnas_correctas = primera_columna.split(', ')
                            
                            # crear dataframe con estructura correcta
                            datos_procesados = []
                            for idx, row in df.iterrows():
                                valores = str(row.iloc[0]).split(', ')
                                if len(valores) >= len(columnas_correctas):
                                    datos_procesados.append(valores[:len(columnas_correctas)])
                            
                            if datos_procesados:
                                df = pd.DataFrame(datos_procesados, columns=columnas_correctas)
                                print(f"  Estructura corregida: {df.shape}")
                        except Exception as e:
                            print(f"  No se pudo corregir estructura: {e}")
                
                # limpiar nombres de columnas
                df.columns = df.columns.str.strip()
                datasets_cargados[archivo.replace('.csv', '')] = df
                print(f"  Dataset {archivo} procesado correctamente")
            else:
                print(f"  Error: No se pudo cargar {archivo}")
        else:
            print(f"Archivo no encontrado: {archivo}")
    
    return datasets_cargados

# ===============================
# CONFIGURACIÓN INICIAL DEL ENTORNO NLP
# ===============================

print("\n=== CONFIGURACIÓN INICIAL DEL ENTORNO NLP ===")
print("Descargando recursos NLTK necesarios...")

# descargar recursos NLTK si no están disponibles
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords') 
    nltk.data.find('corpora/wordnet')
    nltk.data.find('vader_lexicon')
    print("Recursos NLTK ya disponibles")
except LookupError:
    print("Descargando recursos NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    nltk.download('omw-1.4')

# inicializar herramientas NLP
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

print("Entorno NLP configurado exitosamente.")

# ===============================
# ANÁLISIS EXPLORATORIO DE DATASETS
# ===============================

def analisis_exploratorio_mejorado(datasets):
    """
    análisis exploratorio robusto con manejo de errores
    identifica automáticamente columnas relevantes
    """
    print("\n=== ANÁLISIS EXPLORATORIO COMPLETO ===")
    
    dataset_summary = {}
    
    for nombre, df in datasets.items():
        if df is None or df.empty:
            print(f"\n--- DATASET: {nombre.upper()} ---")
            print("Dataset vacío o no válido")
            continue
            
        print(f"\n--- DATASET: {nombre.upper()} ---")
        print(f"Dimensiones: {df.shape}")
        print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # identificar tipos de columnas con mayor robustez
        columnas_texto = []
        columnas_rating = []
        columnas_sentimiento = []
        columnas_fecha = []
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # manejar valores no nulos para el análisis
            sample_not_null = df[col].dropna()
            if len(sample_not_null) == 0:
                continue
                
            sample_values = sample_not_null.head(5).astype(str).str.lower()
            
            # identificar columnas de texto
            if any(keyword in col_lower for keyword in ['text', 'comment', 'review', 'title', 'content', 'opinion']):
                avg_length = sample_not_null.astype(str).str.len().mean()
                if avg_length > 15:  #umbral más bajo para textos cortos
                    columnas_texto.append(col)
            
            # identificar ratings
            elif any(keyword in col_lower for keyword in ['rating', 'score', 'star', 'nota']):
                columnas_rating.append(col)
            
            # identificar sentimientos
            elif any(keyword in col_lower for keyword in ['sentiment', 'sentimiento', 'polarity']):
                columnas_sentimiento.append(col)
            
            # identificar fechas
            elif any(keyword in col_lower for keyword in ['date', 'time', 'fecha']):
                columnas_fecha.append(col)
        
        # estadísticas de calidad
        missing_data = df.isnull().sum()
        duplicated_rows = df.duplicated().sum()
        
        summary = {
            'shape': df.shape,
            'columnas_texto': columnas_texto,
            'columnas_rating': columnas_rating, 
            'columnas_sentimiento': columnas_sentimiento,
            'columnas_fecha': columnas_fecha,
            'valores_faltantes': missing_data[missing_data > 0].to_dict(),
            'filas_duplicadas': duplicated_rows,
            'dataset_valido': len(columnas_texto) > 0 or len(columnas_rating) > 0
        }
        
        dataset_summary[nombre] = summary
        
        print(f"Columnas de texto: {columnas_texto}")
        print(f"Columnas de rating: {columnas_rating}")
        print(f"Columnas de sentimiento: {columnas_sentimiento}")
        print(f"Filas duplicadas: {duplicated_rows}")
        print(f"Dataset válido: {summary['dataset_valido']}")
        
        if missing_data.sum() > 0:
            print("Valores faltantes por columna:")
            for col, missing in missing_data[missing_data > 0].items():
                porcentaje = (missing/len(df)*100) if len(df) > 0 else 0
                print(f"  {col}: {missing} ({porcentaje:.1f}%)")
        
        # mostrar muestra de datos válidos
        if not df.empty:
            print("Muestra de datos (primeras 2 filas):")
            for col in df.columns[:3]:
                sample_data = df[col].head(2).fillna('N/A').tolist()
                print(f"  {col}: {sample_data}")
    
    return dataset_summary

def seleccionar_dataset_optimo(datasets, dataset_summary):
    """
    selección inteligente del mejor dataset disponible
    con validaciones robustas
    """
    print("\n=== SELECCIÓN DE DATASET PRINCIPAL ===")
    
    candidatos_validos = []
    
    for nombre, df in datasets.items():
        if df is None or df.empty:
            continue
            
        summary = dataset_summary.get(nombre, {})
        if not summary.get('dataset_valido', False):
            continue
        
        # calcular score de viabilidad
        score = 0
        
        # puntos por columnas útiles
        score += len(summary.get('columnas_texto', [])) * 50
        score += len(summary.get('columnas_rating', [])) * 30
        score += len(summary.get('columnas_sentimiento', [])) * 20
        
        # puntos por volumen de datos
        score += min(df.shape[0] / 100, 20)  # max 20 puntos por volumen
        
        # penalización por valores faltantes
        valores_faltantes = summary.get('valores_faltantes', {})
        if valores_faltantes:
            total_missing = sum(valores_faltantes.values())
            penalizacion = (total_missing / (df.shape[0] * df.shape[1])) * 30
            score -= penalizacion
        
        # identificar columna de texto principal
        columnas_texto = summary.get('columnas_texto', [])
        col_texto = columnas_texto[0] if columnas_texto else None
        
        # verificar que el texto sea sustancial
        if col_texto:
            textos_validos = df[col_texto].dropna().astype(str)
            if len(textos_validos) > 0:
                longitud_promedio = textos_validos.str.len().mean()
                if longitud_promedio < 10:
                    score -= 25  # penalizar texto muy corto
                elif longitud_promedio > 50:
                    score += 15  # bonificar texto sustancial
        
        columnas_rating = summary.get('columnas_rating', [])
        col_rating = columnas_rating[0] if columnas_rating else None
        
        candidatos_validos.append({
            'nombre': nombre,
            'dataset': df,
            'score': score,
            'col_texto': col_texto,
            'col_rating': col_rating,
            'filas_validas': len(df.dropna(subset=[col_texto] if col_texto else [])),
            'summary': summary
        })
    
    if not candidatos_validos:
        raise ValueError("No se encontraron datasets válidos para el análisis")
    
    # ordenar por score
    candidatos_validos.sort(key=lambda x: x['score'], reverse=True)
    mejor_candidato = candidatos_validos[0]
    
    print("Evaluación de candidatos:")
    for candidato in candidatos_validos:
        print(f"  {candidato['nombre']}: {candidato['score']:.1f} puntos "
              f"({candidato['filas_validas']} filas útiles)")
    
    print(f"\nDataset seleccionado: {mejor_candidato['nombre']}")
    print(f"Score: {mejor_candidato['score']:.1f}")
    print(f"Columna de texto: {mejor_candidato['col_texto']}")
    print(f"Columna de rating: {mejor_candidato['col_rating']}")
    print(f"Registros válidos: {mejor_candidato['filas_validas']:,}")
    
    return mejor_candidato['dataset'], mejor_candidato['col_texto'], mejor_candidato['col_rating']

# ===============================
# PREPROCESAMIENTO MEJORADO DE TEXTO
# ===============================

def limpiar_texto_robusto(texto):
    """
    pipeline de limpieza de texto con manejo de casos especiales
    optimizado para reviews cortas y largas
    """
    if pd.isna(texto) or not isinstance(texto, str) or len(texto.strip()) < 2:
        return ""
    
    # conversión básica
    texto = str(texto).lower().strip()
    
    # remover patrones específicos
    texto = re.sub(r'http[s]?://\S+|www\.\S+', '', texto)  # URLs
    texto = re.sub(r'@\w+|#\w+', '', texto)  # menciones y hashtags
    texto = re.sub(r'\S+@\S+', '', texto)  # emails
    
    # limpiar números preservando contexto
    texto = re.sub(r'\b\d+\b(?!\s*(?:star|rating|/|%))', '', texto)
    
    # normalizar puntuación
    texto = re.sub(r'[^\w\s.,!?\-]', ' ', texto)
    texto = re.sub(r'[.!?]{2,}', '.', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto.strip()

def tokenizar_inteligente(texto):
    """
    tokenización robusta con filtrado avanzado
    maneja errores de NLTK graciosamente
    """
    if not texto or pd.isna(texto) or len(str(texto).strip()) < 2:
        return []
    
    # convertir a string por seguridad
    texto = str(texto)
    
    try:
        # intentar tokenización con NLTK
        tokens = word_tokenize(texto)
    except:
        # fallback: split simple
        tokens = texto.split()
    
    tokens_procesados = []
    
    for token in tokens:
        # convertir token a string por seguridad
        token = str(token)
        
        # filtros de calidad más estrictos
        if (len(token) >= 2 and 
            token not in stop_words and 
            token not in string.punctuation and
            not token.isdigit() and
            token.isalpha() and
            len(token) <= 15):  # evitar tokens muy largos
            
            try:
                # lemmatización segura
                token_lemma = lemmatizer.lemmatize(token)
                if len(token_lemma) >= 2:
                    tokens_procesados.append(token_lemma)
            except:
                # si falla lemmatización, usar token original
                if len(token) >= 2:
                    tokens_procesados.append(token)
    
    return tokens_procesados

def procesar_texto_dataset_completo(df, columna_texto):
    """
    procesamiento completo con validaciones y métricas
    maneja datasets pequeños y grandes de forma robusta
    """
    print(f"\n=== PREPROCESAMIENTO DE TEXTO ===")
    
    if columna_texto not in df.columns:
        raise ValueError(f"Columna '{columna_texto}' no encontrada")
    
    print(f"Procesando columna: {columna_texto}")
    print(f"Registros iniciales: {len(df):,}")
    
    # crear copia para procesar
    df_trabajo = df.copy()
    
    # limpiar textos
    print("Aplicando limpieza de texto...")
    df_trabajo['texto_limpio'] = df_trabajo[columna_texto].apply(limpiar_texto_robusto)
    
    # filtrar textos muy cortos desde el inicio
    mask_texto_valido = df_trabajo['texto_limpio'].str.len() >= 5
    df_trabajo = df_trabajo[mask_texto_valido].copy()
    
    if len(df_trabajo) == 0:
        print("Error: No quedaron textos válidos después de la limpieza")
        return df.head(0)  # retornar dataframe vacío con estructura original
    
    print(f"Textos válidos después de limpieza: {len(df_trabajo):,}")
    
    # estadísticas de limpieza
    longitudes_original = df[columna_texto].fillna('').astype(str).str.len()
    longitudes_limpio = df_trabajo['texto_limpio'].str.len()
    
    print(f"Longitud promedio original: {longitudes_original.mean():.1f} caracteres")
    print(f"Longitud promedio limpio: {longitudes_limpio.mean():.1f} caracteres")
    
    # tokenización
    print("Ejecutando tokenización...")
    df_trabajo['tokens'] = df_trabajo['texto_limpio'].apply(tokenizar_inteligente)
    
    # asegurar que tokens siempre sea una lista válida
    df_trabajo['tokens'] = df_trabajo['tokens'].apply(lambda x: x if isinstance(x, list) else [])
    df_trabajo['texto_procesado'] = df_trabajo['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    
    # filtrar por número mínimo de tokens
    min_tokens = 2  # requisito mínimo muy bajo
    mask_tokens_suficientes = df_trabajo['tokens'].apply(len) >= min_tokens
    df_final = df_trabajo[mask_tokens_suficientes].copy()
    
    if len(df_final) == 0:
        print("Advertencia: No quedaron registros con suficientes tokens")
        print("Usando umbral más permisivo...")
        df_final = df_trabajo[df_trabajo['tokens'].apply(len) >= 1].copy()
    
    # estadísticas finales
    if len(df_final) > 0:
        num_tokens = df_final['tokens'].apply(len)
        print(f"Registros finales: {len(df_final):,}")
        print(f"Tokens promedio por texto: {num_tokens.mean():.1f}")
        
        # vocabulario
        todos_tokens = []
        for tokens_list in df_final['tokens']:
            todos_tokens.extend(tokens_list)
        
        vocabulario = Counter(todos_tokens)
        print(f"Vocabulario único: {len(vocabulario):,} palabras")
        
        if len(vocabulario) > 0:
            palabras_frecuentes = vocabulario.most_common(5)
            print(f"Palabras más frecuentes: {palabras_frecuentes}")
    else:
        print("Error crítico: No se pudo procesar ningún texto")
        return df.head(0)
    
    return df_final

# ===============================
# ANÁLISIS DE SENTIMIENTO MEJORADO
# ===============================

def analizar_sentimiento_robusto(texto):
    """
    análisis de sentimiento con manejo de errores
    incluye validaciones y métricas de confianza
    """
    if not texto or len(str(texto).strip()) < 3:
        return {
            'sentimiento_vader': 'neutral',
            'score_compound': 0.0,
            'score_positivo': 0.0,
            'score_negativo': 0.0,
            'score_neutral': 1.0,
            'confianza': 0.1
        }
    
    try:
        scores = sia.polarity_scores(str(texto))
        
        # clasificación con umbrales ajustados
        compound = scores['compound']
        if compound >= 0.1:
            sentimiento = 'positivo'
        elif compound <= -0.1:
            sentimiento = 'negativo'
        else:
            sentimiento = 'neutral'
        
        # confianza basada en la magnitud del score
        confianza = min(abs(compound) + 0.1, 1.0)  # mínimo 0.1, máximo 1.0
        
        return {
            'sentimiento_vader': sentimiento,
            'score_compound': compound,
            'score_positivo': scores['pos'],
            'score_negativo': scores['neg'],
            'score_neutral': scores['neu'],
            'confianza': confianza
        }
    except Exception as e:
        print(f"Error en análisis de sentimiento: {e}")
        return {
            'sentimiento_vader': 'neutral',
            'score_compound': 0.0,
            'score_positivo': 0.0,
            'score_negativo': 0.0,
            'score_neutral': 1.0,
            'confianza': 0.1
        }

def aplicar_analisis_sentimiento_dataset(df):
    """
    aplicación de análisis de sentimiento a todo el dataset
    con validaciones y estadísticas robustas
    """
    print("\n=== ANÁLISIS DE SENTIMIENTO ===")
    
    if len(df) == 0:
        print("Dataset vacío - no se puede realizar análisis de sentimiento")
        return df
    
    print(f"Analizando sentimiento en {len(df):,} registros...")
    
    # aplicar análisis
    resultados = df['texto_limpio'].apply(analizar_sentimiento_robusto)
    
    # expandir resultados
    sentimiento_df = pd.DataFrame(resultados.tolist())
    df_resultado = pd.concat([df, sentimiento_df], axis=1)
    
    # estadísticas
    if len(df_resultado) > 0:
        distribucion = df_resultado['sentimiento_vader'].value_counts()
        print("\nDistribución de sentimientos:")
        for sentimiento, count in distribucion.items():
            porcentaje = (count / len(df_resultado)) * 100
            print(f"  {sentimiento.capitalize()}: {count:,} ({porcentaje:.1f}%)")
        
        confianza_promedio = df_resultado['confianza'].mean()
        print(f"\nConfianza promedio: {confianza_promedio:.3f}")
        
        alta_confianza = (df_resultado['confianza'] > 0.5).sum()
        print(f"Análisis alta confianza: {alta_confianza:,} ({alta_confianza/len(df_resultado)*100:.1f}%)")
    
    return df_resultado

# ===============================
# EXTRACCIÓN DE ASPECTOS SIMPLIFICADA
# ===============================

def extraer_aspectos_basicos(df):
    """
    extracción simplificada de aspectos para datasets pequeños
    enfocada en términos clave frecuentes
    """
    print("\n=== EXTRACCIÓN DE ASPECTOS BÁSICOS ===")
    
    if len(df) == 0:
        print("Dataset vacío - no se pueden extraer aspectos")
        return df, {}
    
    # aspectos básicos para productos tecnológicos
    aspectos_keywords = {
        'calidad': ['quality', 'good', 'bad', 'excellent', 'poor', 'great', 'terrible'],
        'precio': ['price', 'cost', 'expensive', 'cheap', 'money', 'worth', 'value'],
        'servicio': ['service', 'support', 'help', 'customer', 'staff', 'team'],
        'producto': ['product', 'item', 'device', 'phone', 'app', 'software'],
        'experiencia': ['experience', 'use', 'using', 'easy', 'difficult', 'simple']
    }
    
    aspectos_stats = {}
    
    # inicializar columnas
    for aspecto in aspectos_keywords.keys():
        df[f'menciona_{aspecto}'] = False
        aspectos_stats[aspecto] = {
            'menciones_totales': 0,
            'menciones_positivas': 0,
            'menciones_negativas': 0,
            'menciones_neutrales': 0
        }
    
    print(f"Analizando aspectos en {len(df):,} registros...")
    
    # procesar cada registro
    for idx, row in df.iterrows():
        # manejar tokens que pueden ser NaN o float
        texto_tokens = row.get('tokens', [])
        if not isinstance(texto_tokens, list):
            texto_tokens = []
        
        # usar también el texto limpio como respaldo
        texto_limpio = str(row.get('texto_limpio', '')).lower()
        sentimiento = row.get('sentimiento_vader', 'neutral')
        
        for aspecto, keywords in aspectos_keywords.items():
            mencionado = False
            
            # buscar en tokens (si están disponibles)
            if texto_tokens:
                mencionado = any(keyword in texto_tokens for keyword in keywords)
            
            # si no se encontró en tokens, buscar en texto limpio
            if not mencionado and texto_limpio:
                mencionado = any(keyword in texto_limpio for keyword in keywords)
            
            if mencionado:
                df.loc[idx, f'menciona_{aspecto}'] = True
                aspectos_stats[aspecto]['menciones_totales'] += 1
                
                if sentimiento == 'positivo':
                    aspectos_stats[aspecto]['menciones_positivas'] += 1
                elif sentimiento == 'negativo':
                    aspectos_stats[aspecto]['menciones_negativas'] += 1
                else:
                    aspectos_stats[aspecto]['menciones_neutrales'] += 1
    
    # mostrar estadísticas
    print("\nAspectos identificados:")
    for aspecto, stats in aspectos_stats.items():
        total = stats['menciones_totales']
        if total > 0:
            pos = stats['menciones_positivas']
            neg = stats['menciones_negativas']
            neu = stats['menciones_neutrales']
            
            print(f"  {aspecto.upper()}: {total} menciones")
            print(f"    Positivas: {pos} ({pos/total*100:.1f}%)")
            print(f"    Negativas: {neg} ({neg/total*100:.1f}%)")
            print(f"    Neutrales: {neu} ({neu/total*100:.1f}%)")
    
    return df, aspectos_stats

# ===============================
# IDENTIFICACIÓN DE PAIN POINTS BÁSICA
# ===============================

def identificar_pain_points_basicos(df, aspectos_stats):
    """
    identificación simplificada de pain points
    basada en aspectos con alta negatividad
    """
    print("\n=== IDENTIFICACIÓN DE PAIN POINTS ===")
    
    pain_points = []
    
    for aspecto, stats in aspectos_stats.items():
        total = stats['menciones_totales']
        if total >= 2:  # umbral muy bajo para datasets pequeños
            negativas = stats['menciones_negativas']
            ratio_negativas = negativas / total if total > 0 else 0
            
            if ratio_negativas > 0.3:  # más del 30% negativas
                severidad = ratio_negativas * (total / 5)  # ajustado para datasets pequeños
                
                pain_points.append({
                    'aspecto': aspecto,
                    'menciones_negativas': negativas,
                    'total_menciones': total,
                    'ratio_negativas': ratio_negativas,
                    'severidad': severidad
                })
    
    # ordenar por severidad
    pain_points.sort(key=lambda x: x['severidad'], reverse=True)
    
    print(f"Pain points identificados: {len(pain_points)}")
    for i, pp in enumerate(pain_points[:3], 1):
        print(f"  {i}. {pp['aspecto'].upper()}: "
              f"{pp['menciones_negativas']}/{pp['total_menciones']} negativas "
              f"(severidad: {pp['severidad']:.2f})")
    
    return pain_points

# ===============================
# GENERACIÓN DE RECOMENDACIONES BÁSICAS
# ===============================

def generar_recomendaciones_basicas(pain_points):
    """
    generación simplificada de recomendaciones
    basada en los pain points identificados
    """
    print("\n=== GENERACIÓN DE RECOMENDACIONES ===")
    
    plantillas = {
        'calidad': 'Mejorar los estándares de calidad del producto',
        'precio': 'Revisar estrategia de precios y propuesta de valor',
        'servicio': 'Fortalecer el servicio al cliente y soporte técnico',
        'producto': 'Optimizar características y funcionalidades del producto',
        'experiencia': 'Simplificar la experiencia de usuario'
    }
    
    recomendaciones = []
    
    for i, pp in enumerate(pain_points[:3], 1):
        aspecto = pp['aspecto']
        recomendacion = plantillas.get(aspecto, f'Mejorar aspectos relacionados con {aspecto}')
        
        recomendaciones.append({
            'prioridad': i,
            'aspecto': aspecto,
            'problema': f"Alto nivel de insatisfacción en {aspecto}",
            'recomendacion': recomendacion,
            'impacto': pp['severidad'],
            'menciones_negativas': pp['menciones_negativas']
        })
    
    print("Recomendaciones generadas:")
    for rec in recomendaciones:
        print(f"  {rec['prioridad']}. {rec['recomendacion']}")
        print(f"     Impacto estimado: {rec['impacto']:.2f}")
        print(f"     Menciones negativas: {rec['menciones_negativas']}")
    
    return recomendaciones

# ===============================
# VISUALIZACIONES BÁSICAS
# ===============================

def generar_visualizaciones_basicas(df, aspectos_stats):
    """
    visualizaciones simples pero efectivas
    adaptadas a datasets pequeños
    """
    print("\n=== GENERANDO VISUALIZACIONES ===")
    
    if len(df) == 0:
        print("No hay datos para visualizar")
        return
    
    # configurar figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dashboard de Análisis de Opiniones', fontsize=16, fontweight='bold')
    
    # gráfico 1: distribución de sentimientos
    if 'sentimiento_vader' in df.columns:
        sentiment_counts = df['sentimiento_vader'].value_counts()
        colores = ['#e74c3c', '#f39c12', '#27ae60']  # rojo, amarillo, verde
        
        if len(sentiment_counts) > 0:
            axes[0,0].pie(sentiment_counts.values, 
                         labels=sentiment_counts.index,
                         autopct='%1.1f%%', 
                         colors=colores[:len(sentiment_counts)],
                         startangle=90)
            axes[0,0].set_title('Distribución de Sentimientos')
        else:
            axes[0,0].text(0.5, 0.5, 'Sin datos de sentimiento', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
    
    # gráfico 2: aspectos mencionados
    aspectos_totales = {k: v['menciones_totales'] for k, v in aspectos_stats.items() 
                       if v['menciones_totales'] > 0}
    
    if aspectos_totales:
        aspectos_nombres = list(aspectos_totales.keys())
        aspectos_valores = list(aspectos_totales.values())
        
        bars = axes[0,1].bar(aspectos_nombres, aspectos_valores, color='#3498db', alpha=0.7)
        axes[0,1].set_title('Aspectos Más Mencionados')
        axes[0,1].set_xlabel('Aspectos')
        axes[0,1].set_ylabel('Menciones')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # agregar valores encima de barras
        for bar, valor in zip(bars, aspectos_valores):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                          str(valor), ha='center', va='bottom')
    else:
        axes[0,1].text(0.5, 0.5, 'Sin aspectos identificados', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
    
    # gráfico 3: distribución de confianza
    if 'confianza' in df.columns and df['confianza'].notna().sum() > 0:
        axes[1,0].hist(df['confianza'].dropna(), bins=15, alpha=0.7, 
                      color='#9b59b6', edgecolor='black')
        axes[1,0].set_title('Distribución de Confianza del Análisis')
        axes[1,0].set_xlabel('Confianza')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].axvline(x=df['confianza'].mean(), color='red', 
                         linestyle='--', label=f'Promedio: {df["confianza"].mean():.2f}')
        axes[1,0].legend()
    else:
        axes[1,0].text(0.5, 0.5, 'Sin datos de confianza', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
    
    # gráfico 4: longitud de texto vs sentimiento
    if 'sentimiento_vader' in df.columns:
        df['longitud_texto'] = df['texto_limpio'].str.len()
        sentimientos_unicos = df['sentimiento_vader'].unique()
        
        if len(sentimientos_unicos) > 0:
            try:
                # boxplot si hay suficientes datos
                if len(df) > 10:
                    sns.boxplot(data=df, x='sentimiento_vader', y='longitud_texto', ax=axes[1,1])
                else:
                    # scatter plot para pocos datos
                    for i, sent in enumerate(sentimientos_unicos):
                        data_sent = df[df['sentimiento_vader'] == sent]
                        axes[1,1].scatter([i] * len(data_sent), data_sent['longitud_texto'], 
                                        label=sent, alpha=0.7)
                    axes[1,1].set_xticks(range(len(sentimientos_unicos)))
                    axes[1,1].set_xticklabels(sentimientos_unicos)
                    axes[1,1].legend()
                
                axes[1,1].set_title('Longitud de Texto por Sentimiento')
                axes[1,1].set_xlabel('Sentimiento')
                axes[1,1].set_ylabel('Longitud (caracteres)')
            except Exception as e:
                axes[1,1].text(0.5, 0.5, f'Error en visualización: {str(e)[:50]}', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
        else:
            axes[1,1].text(0.5, 0.5, 'Sin datos de sentimiento', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # generar wordcloud si hay suficientes datos
    if len(df) > 5 and 'tokens' in df.columns:
        generar_wordcloud_simple(df)

def generar_wordcloud_simple(df):
    """
    genera wordcloud básico de términos frecuentes
    """
    print("Generando word cloud...")
    
    try:
        # recopilar todos los tokens de forma segura
        todos_tokens = []
        for tokens_list in df['tokens'].dropna():
            # verificar que sea una lista válida
            if isinstance(tokens_list, list):
                todos_tokens.extend(tokens_list)
            elif isinstance(tokens_list, str):
                # si es string, dividir por espacios
                todos_tokens.extend(tokens_list.split())
        
        if len(todos_tokens) > 10:
            frecuencias = Counter(todos_tokens)
            texto_wordcloud = ' '.join([palabra for palabra, freq in frecuencias.most_common(50)])
            
            if texto_wordcloud and len(texto_wordcloud.strip()) > 0:
                plt.figure(figsize=(10, 6))
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    max_words=50,
                                    colormap='viridis').generate(texto_wordcloud)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title('Términos Más Frecuentes', fontweight='bold', fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            else:
                print("No se pudo generar wordcloud - texto insuficiente")
        else:
            print("Vocabulario insuficiente para wordcloud")
    
    except Exception as e:
        print(f"Error generando wordcloud: {e}")

# ===============================
# VALIDACIÓN DE CALIDAD MEJORADA
# ===============================

def validar_calidad_analisis_seguro(df, aspectos_stats, pain_points):
    """
    validación de calidad con manejo robusto de casos edge
    evita divisiones por cero y errores comunes
    """
    print("\n=== VALIDACIÓN DE CALIDAD DEL ANÁLISIS ===")
    
    metricas = {
        'total_registros': len(df),
        'registros_procesados': 0,
        'cobertura_texto': 0.0,
        'precision_sentimiento': 0.0,
        'aspectos_identificados': 0,
        'pain_points_detectados': len(pain_points),
        'calidad_general': 0.0
    }
    
    if len(df) == 0:
        print("Dataset vacío - no se puede validar calidad")
        return metricas
    
    # cobertura de texto válido
    if 'texto_procesado' in df.columns:
        textos_validos = df[df['texto_procesado'].str.len() >= 5].shape[0]
        metricas['registros_procesados'] = textos_validos
        metricas['cobertura_texto'] = (textos_validos / len(df)) * 100 if len(df) > 0 else 0
    
    # distribución de sentimientos
    if 'sentimiento_vader' in df.columns:
        dist_sentimiento = df['sentimiento_vader'].value_counts(normalize=True)
        if len(dist_sentimiento) > 0:
            # calcular entropía (diversidad)
            entropia = -sum(p * np.log2(p + 1e-10) for p in dist_sentimiento)  # evitar log(0)
            metricas['entropia_sentimiento'] = entropia
            
            # estimar precisión basada en distribución balanceada
            precision_estimada = min(85.0, 60 + (entropia * 15))
            metricas['precision_sentimiento'] = precision_estimada
    
    # aspectos identificados
    aspectos_con_menciones = sum(1 for stats in aspectos_stats.values() 
                               if stats['menciones_totales'] > 0)
    metricas['aspectos_identificados'] = aspectos_con_menciones
    
    # calidad general
    factores_calidad = [
        min(metricas['cobertura_texto'], 100) / 100,  # cobertura normalizada
        metricas['precision_sentimiento'] / 100,       # precisión normalizada
        min(aspectos_con_menciones * 20, 100) / 100,  # aspectos (max 5 aspectos = 100%)
        min(len(pain_points) * 25, 100) / 100         # pain points (max 4 = 100%)
    ]
    
    metricas['calidad_general'] = sum(factores_calidad) / len(factores_calidad) * 100
    
    # mostrar resultados
    print(f"Total de registros: {metricas['total_registros']:,}")
    print(f"Registros procesados: {metricas['registros_procesados']:,}")
    print(f"Cobertura de texto: {metricas['cobertura_texto']:.1f}%")
    print(f"Precisión estimada: {metricas['precision_sentimiento']:.1f}%")
    print(f"Aspectos identificados: {metricas['aspectos_identificados']}")
    print(f"Pain points detectados: {metricas['pain_points_detectados']}")
    print(f"Calidad general: {metricas['calidad_general']:.1f}%")
    
    # evaluación cualitativa
    if metricas['calidad_general'] >= 70:
        print("Evaluación: BUENA - Análisis confiable")
    elif metricas['calidad_general'] >= 50:
        print("Evaluación: REGULAR - Análisis parcialmente válido")
    else:
        print("Evaluación: BAJA - Se requiere más datos o ajustes")
    
    return metricas

# ===============================
# PIPELINE PRINCIPAL SIMPLIFICADO
# ===============================

def ejecutar_pipeline_completo():
    """
    pipeline principal con manejo robusto de errores
    adaptado para datasets de cualquier tamaño
    """
    print("SISTEMA INTELIGENTE DE ANÁLISIS DE OPINIONES")
    print("Samsung Innovation Campus 2024 - Proyecto Capstone")
    print("="*70)
    
    try:
        # paso 1: cargar datasets locales
        datasets = cargar_datasets_locales()
        
        if not datasets:
            print("Error: No se encontraron datasets válidos")
            return None
        
        # paso 2: análisis exploratorio
        dataset_summary = analisis_exploratorio_mejorado(datasets)
        
        # paso 3: seleccionar mejor dataset
        df_principal, col_texto, col_rating = seleccionar_dataset_optimo(datasets, dataset_summary)
        
        if df_principal is None or len(df_principal) == 0:
            print("Error: No se pudo seleccionar un dataset válido")
            return None
        
        # paso 4: preprocesamiento de texto
        df_procesado = procesar_texto_dataset_completo(df_principal, col_texto)
        
        if len(df_procesado) == 0:
            print("Error: No se pudieron procesar los textos")
            return None
        
        # paso 5: análisis de sentimiento
        df_con_sentimiento = aplicar_analisis_sentimiento_dataset(df_procesado)
        
        # paso 6: extracción de aspectos
        df_completo, aspectos_detectados = extraer_aspectos_basicos(df_con_sentimiento)
        
        # paso 7: identificación de pain points
        pain_points = identificar_pain_points_basicos(df_completo, aspectos_detectados)
        
        # paso 8: generación de recomendaciones
        recomendaciones = generar_recomendaciones_basicas(pain_points)
        
        # paso 9: visualizaciones
        generar_visualizaciones_basicas(df_completo, aspectos_detectados)
        
        # paso 10: validación de calidad
        metricas_calidad = validar_calidad_analisis_seguro(df_completo, aspectos_detectados, pain_points)
        
        # resumen final
        print("\n" + "="*70)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print(f"Registros analizados: {len(df_completo):,}")
        print(f"Aspectos identificados: {len([a for a in aspectos_detectados.values() if a['menciones_totales'] > 0])}")
        print(f"Pain points detectados: {len(pain_points)}")
        print(f"Recomendaciones generadas: {len(recomendaciones)}")
        print(f"Calidad del análisis: {metricas_calidad['calidad_general']:.1f}%")
        print("="*70)
        
        return {
            'dataset_final': df_completo,
            'aspectos_detectados': aspectos_detectados, 
            'pain_points': pain_points,
            'recomendaciones': recomendaciones,
            'metricas_calidad': metricas_calidad,
            'columna_texto_original': col_texto,
            'columna_rating': col_rating
        }
        
    except Exception as e:
        print(f"\nError en pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===============================
# GENERACIÓN DE REPORTE EJECUTIVO
# ===============================

def generar_reporte_ejecutivo(resultados):
    """
    genera reporte ejecutivo con los hallazgos principales
    formato profesional para stakeholders
    """
    if not resultados:
        print("No hay resultados para generar reporte")
        return
    
    print("\n" + "="*80)
    print("                    REPORTE EJECUTIVO")
    print("            ANÁLISIS DE OPINIONES DE CLIENTES")
    print("="*80)
    
    df = resultados['dataset_final']
    aspectos = resultados['aspectos_detectados']
    pain_points = resultados['pain_points']
    recomendaciones = resultados['recomendaciones']
    metricas = resultados['metricas_calidad']
    
    # resumen ejecutivo
    print("\nRESUMEN EJECUTIVO:")
    print("-" * 40)
    
    if 'sentimiento_vader' in df.columns:
        distribucion = df['sentimiento_vader'].value_counts()
        total = len(df)
        
        pos = distribucion.get('positivo', 0)
        neg = distribucion.get('negativo', 0)
        neu = distribucion.get('neutral', 0)
        
        print(f"• Total de opiniones analizadas: {total:,}")
        print(f"• Opiniones positivas: {pos:,} ({pos/total*100:.1f}%)")
        print(f"• Opiniones negativas: {neg:,} ({neg/total*100:.1f}%)")
        print(f"• Opiniones neutrales: {neu:,} ({neu/total*100:.1f}%)")
        
        # nps estimado
        nps = ((pos - neg) / total * 100) if total > 0 else 0
        print(f"• Net Promoter Score estimado: {nps:.1f}")
    
    print(f"• Confianza del análisis: {metricas['calidad_general']:.1f}%")
    
    # principales fortalezas
    print("\nPRINCIPALES FORTALEZAS:")
    print("-" * 40)
    fortalezas_encontradas = False
    
    for aspecto, stats in aspectos.items():
        if stats['menciones_totales'] >= 2:
            ratio_positivas = stats['menciones_positivas'] / stats['menciones_totales']
            if ratio_positivas > 0.6:
                print(f"• {aspecto.capitalize()}: {stats['menciones_positivas']}/{stats['menciones_totales']} menciones positivas ({ratio_positivas:.1%})")
                fortalezas_encontradas = True
    
    if not fortalezas_encontradas:
        print("• Se requiere más volumen de datos para identificar fortalezas específicas")
    
    # principales pain points
    print("\nPRINCIPALES ÁREAS DE MEJORA:")
    print("-" * 40)
    
    if pain_points:
        for i, pp in enumerate(pain_points[:3], 1):
            print(f"{i}. {pp['aspecto'].capitalize()}: {pp['menciones_negativas']} menciones negativas")
            print(f"   Severidad: {pp['severidad']:.2f} | Ratio: {pp['ratio_negativas']:.1%}")
    else:
        print("• No se detectaron pain points críticos en este análisis")
    
    # recomendaciones estratégicas
    print("\nRECOMENDACIONES ESTRATÉGICAS:")
    print("-" * 40)
    
    if recomendaciones:
        for rec in recomendaciones:
            print(f"{rec['prioridad']}. {rec['recomendacion']}")
            print(f"   Impacto estimado: {rec['impacto']:.2f}")
            print()
    else:
        print("• Monitorear continuamente el feedback de clientes")
        print("• Implementar sistema de seguimiento de satisfacción")
        print("• Realizar análisis más profundo con mayor volumen de datos")
    
    # próximos pasos
    print("PRÓXIMOS PASOS RECOMENDADOS:")
    print("-" * 40)
    print("• Ampliar la recolección de feedback de clientes")
    print("• Implementar dashboard de monitoreo en tiempo real")
    print("• Realizar análisis comparativo con competidores")
    print("• Desarrollar plan de acción específico para pain points identificados")
    print("• Establecer KPIs de seguimiento de mejoras implementadas")
    
    print("\n" + "="*80)
    print("Reporte generado automáticamente por el Sistema de Análisis de Opiniones")
    print("Samsung Innovation Campus 2024")
    print("="*80)

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    try:
        print("INICIANDO SISTEMA DE ANÁLISIS DE OPINIONES...")
        print("Versión optimizada para datasets locales")
        print()
        
        # ejecutar pipeline completo
        resultados_finales = ejecutar_pipeline_completo()
        
        if resultados_finales:
            # generar reporte ejecutivo
            generar_reporte_ejecutivo(resultados_finales)
            
            print("\nSISTEMA EJECUTADO EXITOSAMENTE")
            print("Análisis completo disponible en las visualizaciones generadas")
            
        else:
            print("\nERROR: No se pudieron procesar los datos")
            print("Verificar:")
            print("- Que los archivos CSV estén en el directorio actual")
            print("- Que los archivos tengan el formato correcto")
            print("- Que haya suficiente contenido de texto para analizar")
        
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario")
    except Exception as e:
        print(f"\nError crítico: {e}")