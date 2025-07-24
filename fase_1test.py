# Sistema Inteligente de Análisis de Opiniones y Generación Automática de Recomendaciones
# FASE 1 MEJORADA - Análisis por cada dataset individual
# Proyecto Capstone - Samsung Innovation Campus 2024

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

print("Sistema de análisis de opiniones - Pipeline NLP avanzado")
print("Análisis individual por dataset - Metodología Samsung Innovation Campus")
print("-"*75)

# ===============================
# CONFIGURACIÓN INICIAL DEL ENTORNO NLP
# ===============================

print("\n=== Configuración inicial del entorno NLP ===")
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
# CARGA MEJORADA DE DATASETS
# ===============================

def cargar_dataset_amazon_mejorado(archivo_path='Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv'):
    """
    función específica para cargar el dataset de Amazon con manejo correcto de columnas
    """
    print(f"Procesando dataset de Amazon: {archivo_path}")
    
    if not os.path.exists(archivo_path):
        print(f"Archivo {archivo_path} no encontrado")
        return None, "amazon_reviews"
    
    try:
        # intentar cargar con diferentes encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df_amazon = pd.read_csv(archivo_path, encoding=encoding, low_memory=False)
                print(f"  Dataset cargado con encoding {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            print("  No se pudo cargar con ningún encoding")
            return None, "amazon_reviews"
        
        print(f"  Dimensiones originales: {df_amazon.shape}")
        
        # verificar columnas clave y mapear
        mapeo_columnas = {
            'reviews.text': 'texto_opinion',
            'reviews.rating': 'rating', 
            'reviews.title': 'titulo_review',
            'name': 'producto',
            'reviews.date': 'fecha',
            'brand': 'marca'
        }
        
        # renombrar columnas que existan
        for col_original, col_nuevo in mapeo_columnas.items():
            if col_original in df_amazon.columns:
                df_amazon[col_nuevo] = df_amazon[col_original]
        
        # verificar columna principal de texto
        if 'texto_opinion' not in df_amazon.columns:
            print("  Error: No se encontró reviews.text")
            return None, "amazon_reviews"
        
        # limpiar datos básicos
        df_amazon = df_amazon.dropna(subset=['texto_opinion'])
        df_amazon = df_amazon[df_amazon['texto_opinion'].str.len() > 10]
        
        # convertir rating si existe
        if 'rating' in df_amazon.columns:
            df_amazon['rating'] = pd.to_numeric(df_amazon['rating'], errors='coerce')
        
        print(f"  Dataset procesado exitosamente: {len(df_amazon):,} registros válidos")
        return df_amazon, "amazon_reviews"
        
    except Exception as e:
        print(f"  Error procesando Amazon dataset: {e}")
        return None, "amazon_reviews"

def cargar_todos_los_datasets():
    """
    carga todos los datasets disponibles con manejo mejorado
    """
    print("\n=== Carga de datasets ===")
    print("Identificando y cargando todos los datasets disponibles...")
    
    datasets_cargados = {}
    
    # 1. dataset de Amazon (prioritario)
    df_amazon, nombre_amazon = cargar_dataset_amazon_mejorado()
    if df_amazon is not None:
        datasets_cargados[nombre_amazon] = {
            'dataframe': df_amazon,
            'columna_texto': 'texto_opinion',
            'columna_rating': 'rating' if 'rating' in df_amazon.columns else None,
            'fuente': 'Amazon Consumer Reviews',
            'descripcion': 'Reviews de productos Amazon'
        }
    
    # 2. datasets CSV regulares  
    archivos_csv = ['sentimentanalysis.csv', 'redmi6.csv']
    
    for archivo in archivos_csv:
        if os.path.exists(archivo):
            print(f"\nProcesando: {archivo}")
            
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(archivo, encoding=encoding)
                    print(f"  Cargado con encoding {encoding}")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            
            if df is not None:
                # corregir estructura problemática de sentimentanalysis
                if 'sentimentanalysis' in archivo and df.shape[1] == 1:
                    primera_columna = df.columns[0]
                    if ',' in primera_columna:
                        print("  Corrigiendo estructura de columnas...")
                        try:
                            columnas_correctas = primera_columna.split(', ')
                            datos_procesados = []
                            
                            for idx, row in df.iterrows():
                                valores = str(row.iloc[0]).split(', ')
                                if len(valores) >= len(columnas_correctas):
                                    datos_procesados.append(valores[:len(columnas_correctas)])
                            
                            if datos_procesados:
                                df = pd.DataFrame(datos_procesados, columns=columnas_correctas)
                                print(f"  Estructura corregida: {df.shape}")
                        except Exception as e:
                            print(f"  No se pudo corregir: {e}")
                
                # limpiar nombres de columnas
                df.columns = df.columns.str.strip()
                
                # identificar columnas relevantes
                col_texto = identificar_columna_texto(df)
                col_rating = identificar_columna_rating(df)
                
                if col_texto:
                    nombre_dataset = archivo.replace('.csv', '')
                    datasets_cargados[nombre_dataset] = {
                        'dataframe': df,
                        'columna_texto': col_texto,
                        'columna_rating': col_rating,
                        'fuente': archivo,
                        'descripcion': f'Dataset local: {archivo}'
                    }
                    print(f"  Dataset válido agregado: {len(df)} registros")
                else:
                    print(f"  Dataset {archivo} no tiene columna de texto válida")
    
    print(f"\nTotal de datasets cargados: {len(datasets_cargados)}")
    for nombre, info in datasets_cargados.items():
        print(f"  {nombre}: {len(info['dataframe']):,} registros")
    
    return datasets_cargados

def identificar_columna_texto(df):
    """
    identifica la mejor columna de texto en un dataframe
    """
    candidatos_texto = []
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(keyword in col_lower for keyword in ['text', 'comment', 'review', 'title', 'content', 'opinion']):
            sample_not_null = df[col].dropna()
            if len(sample_not_null) > 0:
                avg_length = sample_not_null.astype(str).str.len().mean()
                if avg_length > 10:
                    candidatos_texto.append((col, avg_length))
    
    if candidatos_texto:
        # retornar la columna con mayor longitud promedio
        candidatos_texto.sort(key=lambda x: x[1], reverse=True)
        return candidatos_texto[0][0]
    
    return None

def identificar_columna_rating(df):
    """
    identifica columna de rating si existe
    """
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(keyword in col_lower for keyword in ['rating', 'score', 'star', 'nota']):
            return col
    return None

# ===============================
# PIPELINE DE ANÁLISIS POR DATASET
# ===============================

def limpiar_texto_robusto(texto):
    """
    pipeline de limpieza de texto optimizado
    """
    if pd.isna(texto) or not isinstance(texto, str) or len(texto.strip()) < 2:
        return ""
    
    texto = str(texto).lower().strip()
    
    # remover patrones específicos
    texto = re.sub(r'http[s]?://\S+|www\.\S+', '', texto)  # URLs
    texto = re.sub(r'@\w+|#\w+', '', texto)  # menciones y hashtags
    texto = re.sub(r'\S+@\S+', '', texto)  # emails
    texto = re.sub(r'\b\d+\b(?!\s*(?:star|rating|/|%))', '', texto)  # números sin contexto
    
    # normalizar puntuación
    texto = re.sub(r'[^\w\s.,!?\-]', ' ', texto)
    texto = re.sub(r'[.!?]{2,}', '.', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto.strip()

def tokenizar_inteligente(texto):
    """
    tokenización con filtrado avanzado
    """
    if not texto or pd.isna(texto) or len(str(texto).strip()) < 2:
        return []
    
    texto = str(texto)
    
    try:
        tokens = word_tokenize(texto)
    except:
        tokens = texto.split()
    
    tokens_procesados = []
    
    for token in tokens:
        token = str(token)
        
        if (len(token) >= 2 and 
            token not in stop_words and 
            token not in string.punctuation and
            not token.isdigit() and
            token.isalpha() and
            len(token) <= 15):
            
            try:
                token_lemma = lemmatizer.lemmatize(token)
                if len(token_lemma) >= 2:
                    tokens_procesados.append(token_lemma)
            except:
                if len(token) >= 2:
                    tokens_procesados.append(token)
    
    return tokens_procesados

def analizar_sentimiento_robusto(texto):
    """
    análisis de sentimiento con métricas de confianza
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
        compound = scores['compound']
        
        if compound >= 0.1:
            sentimiento = 'positivo'
        elif compound <= -0.1:
            sentimiento = 'negativo'
        else:
            sentimiento = 'neutral'
        
        confianza = min(abs(compound) + 0.1, 1.0)
        
        return {
            'sentimiento_vader': sentimiento,
            'score_compound': compound,
            'score_positivo': scores['pos'],
            'score_negativo': scores['neg'],
            'score_neutral': scores['neu'],
            'confianza': confianza
        }
    except Exception as e:
        return {
            'sentimiento_vader': 'neutral',
            'score_compound': 0.0,
            'score_positivo': 0.0,
            'score_negativo': 0.0,
            'score_neutral': 1.0,
            'confianza': 0.1
        }

def extraer_aspectos_basicos(df_procesado):
    """
    extracción de aspectos básicos para productos/servicios
    """
    aspectos_keywords = {
        'calidad': ['quality', 'good', 'bad', 'excellent', 'poor', 'great', 'terrible', 'amazing', 'awful'],
        'precio': ['price', 'cost', 'expensive', 'cheap', 'money', 'worth', 'value', 'affordable'],
        'servicio': ['service', 'support', 'help', 'customer', 'staff', 'team', 'assistance'],
        'producto': ['product', 'item', 'device', 'phone', 'app', 'software', 'features'],
        'experiencia': ['experience', 'use', 'using', 'easy', 'difficult', 'simple', 'user'],
        'entrega': ['delivery', 'shipping', 'arrived', 'fast', 'slow', 'package'],
        'diseño': ['design', 'look', 'appearance', 'beautiful', 'ugly', 'style']
    }
    
    aspectos_stats = {}
    
    # inicializar contadores
    for aspecto in aspectos_keywords.keys():
        df_procesado[f'menciona_{aspecto}'] = False
        aspectos_stats[aspecto] = {
            'menciones_totales': 0,
            'menciones_positivas': 0,
            'menciones_negativas': 0,
            'menciones_neutrales': 0
        }
    
    # procesar cada registro
    for idx, row in df_procesado.iterrows():
        texto_tokens = row.get('tokens', [])
        if not isinstance(texto_tokens, list):
            texto_tokens = []
        
        texto_limpio = str(row.get('texto_limpio', '')).lower()
        sentimiento = row.get('sentimiento_vader', 'neutral')
        
        for aspecto, keywords in aspectos_keywords.items():
            mencionado = False
            
            # buscar en tokens
            if texto_tokens:
                mencionado = any(keyword in texto_tokens for keyword in keywords)
            
            # buscar en texto limpio como respaldo
            if not mencionado and texto_limpio:
                mencionado = any(keyword in texto_limpio for keyword in keywords)
            
            if mencionado:
                df_procesado.loc[idx, f'menciona_{aspecto}'] = True
                aspectos_stats[aspecto]['menciones_totales'] += 1
                
                if sentimiento == 'positivo':
                    aspectos_stats[aspecto]['menciones_positivas'] += 1
                elif sentimiento == 'negativo':
                    aspectos_stats[aspecto]['menciones_negativas'] += 1
                else:
                    aspectos_stats[aspecto]['menciones_neutrales'] += 1
    
    return df_procesado, aspectos_stats

def identificar_pain_points(aspectos_stats):
    """
    identificación de pain points basada en negatividad
    """
    pain_points = []
    
    for aspecto, stats in aspectos_stats.items():
        if stats['menciones_totales'] >= 2:
            ratio_negativas = stats['menciones_negativas'] / stats['menciones_totales']
            if ratio_negativas > 0.3:  # más del 30% negativas
                severidad = ratio_negativas * (stats['menciones_totales'] / 5)
                
                pain_points.append({
                    'aspecto': aspecto,
                    'menciones_negativas': stats['menciones_negativas'],
                    'total_menciones': stats['menciones_totales'],
                    'ratio_negativas': ratio_negativas,
                    'severidad': severidad
                })
    
    pain_points.sort(key=lambda x: x['severidad'], reverse=True)
    return pain_points

def generar_recomendaciones_por_dataset(pain_points, nombre_dataset):
    """
    genera recomendaciones específicas basadas en pain points
    """
    plantillas = {
        'calidad': 'Mejorar los estándares de control de calidad',
        'precio': 'Revisar estrategia de precios y comunicación de valor',
        'servicio': 'Fortalecer capacidades de atención al cliente',
        'producto': 'Optimizar características y funcionalidades',
        'experiencia': 'Simplificar y mejorar la experiencia de usuario',
        'entrega': 'Optimizar procesos de entrega y comunicación',
        'diseño': 'Mejorar aspectos de diseño y presentación'
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
            'menciones_negativas': pp['menciones_negativas'],
            'dataset_origen': nombre_dataset
        })
    
    return recomendaciones

def procesar_dataset_individual(nombre_dataset, info_dataset):
    """
    aplica todo el pipeline de análisis a un dataset individual
    """
    print(f"\n{'='*80}")
    print(f"PROCESANDO DATASET: {nombre_dataset.upper()}")
    print(f"Fuente: {info_dataset['descripcion']}")
    print(f"{'='*80}")
    
    df = info_dataset['dataframe']
    col_texto = info_dataset['columna_texto']
    col_rating = info_dataset['columna_rating']
    
    print(f"Registros iniciales: {len(df):,}")
    print(f"Columna de texto: {col_texto}")
    print(f"Columna de rating: {col_rating}")
    
    # paso 1: preprocesamiento de texto
    print(f"\n--- Preprocesamiento de texto ---")
    
    df_trabajo = df.copy()
    
    # limpiar textos
    df_trabajo['texto_limpio'] = df_trabajo[col_texto].apply(limpiar_texto_robusto)
    df_trabajo = df_trabajo[df_trabajo['texto_limpio'].str.len() >= 5]
    
    if len(df_trabajo) == 0:
        print("Error: No quedaron textos válidos después de la limpieza")
        return None
    
    print(f"Textos válidos después de limpieza: {len(df_trabajo):,}")
    
    # tokenizar
    df_trabajo['tokens'] = df_trabajo['texto_limpio'].apply(tokenizar_inteligente)
    df_trabajo['texto_procesado'] = df_trabajo['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    
    # filtrar por tokens mínimos
    df_trabajo = df_trabajo[df_trabajo['tokens'].apply(len) >= 1]
    
    if len(df_trabajo) == 0:
        print("Error: No quedaron registros con tokens válidos")
        return None
    
    print(f"Registros finales: {len(df_trabajo):,}")
    
    # vocabulario
    todos_tokens = []
    for tokens_list in df_trabajo['tokens']:
        if isinstance(tokens_list, list):
            todos_tokens.extend(tokens_list)
    
    vocabulario = Counter(todos_tokens)
    print(f"Vocabulario único: {len(vocabulario):,} palabras")
    
    # paso 2: análisis de sentimiento
    print(f"\n--- Análisis de sentimiento ---")
    
    resultados_sentiment = df_trabajo['texto_limpio'].apply(analizar_sentimiento_robusto)
    sentimiento_df = pd.DataFrame(resultados_sentiment.tolist())
    df_completo = pd.concat([df_trabajo, sentimiento_df], axis=1)
    
    # estadísticas de sentimiento
    distribucion = df_completo['sentimiento_vader'].value_counts()
    print("Distribución de sentimientos:")
    for sent, count in distribucion.items():
        print(f"  {sent.capitalize()}: {count:,} ({count/len(df_completo)*100:.1f}%)")
    
    confianza_promedio = df_completo['confianza'].mean()
    print(f"Confianza promedio: {confianza_promedio:.3f}")
    
    # paso 3: extracción de aspectos
    print(f"\n--- Extracción de aspectos ---")
    
    df_final, aspectos_stats = extraer_aspectos_basicos(df_completo)
    
    print("Aspectos identificados:")
    for aspecto, stats in aspectos_stats.items():
        if stats['menciones_totales'] > 0:
            print(f"  {aspecto}: {stats['menciones_totales']} menciones "
                  f"({stats['menciones_positivas']} pos, {stats['menciones_negativas']} neg)")
    
    # paso 4: identificación de pain points
    print(f"\n--- Identificación de pain points ---")
    
    pain_points = identificar_pain_points(aspectos_stats)
    
    print(f"Pain points identificados: {len(pain_points)}")
    for i, pp in enumerate(pain_points[:3], 1):
        print(f"  {i}. {pp['aspecto']}: {pp['menciones_negativas']}/{pp['total_menciones']} "
              f"negativas (severidad: {pp['severidad']:.2f})")
    
    # paso 5: generación de recomendaciones
    print(f"\n--- Generación de recomendaciones ---")
    
    recomendaciones = generar_recomendaciones_por_dataset(pain_points, nombre_dataset)
    
    print("Recomendaciones generadas:")
    for rec in recomendaciones:
        print(f"  {rec['prioridad']}. {rec['recomendacion']}")
        print(f"     Impacto estimado: {rec['impacto']:.2f}")
    
    # paso 6: métricas de calidad
    print(f"\n--- Métricas de calidad ---")
    
    # NPS estimado
    total = len(df_final)
    promotores = distribucion.get('positivo', 0)
    detractores = distribucion.get('negativo', 0)
    nps = ((promotores - detractores) / total * 100) if total > 0 else 0
    
    print(f"NPS estimado: {nps:.1f}")
    print(f"Promotores: {promotores:,} ({promotores/total*100:.1f}%)")
    print(f"Detractores: {detractores:,} ({detractores/total*100:.1f}%)")
    
    # calidad general
    cobertura_texto = (len(df_final) / len(df)) * 100
    aspectos_identificados = len([a for a in aspectos_stats.values() if a['menciones_totales'] > 0])
    
    factores_calidad = [
        min(cobertura_texto, 100) / 100,
        confianza_promedio,
        min(aspectos_identificados * 20, 100) / 100,
        min(len(pain_points) * 25, 100) / 100
    ]
    
    calidad_general = sum(factores_calidad) / len(factores_calidad) * 100
    
    print(f"Cobertura de texto: {cobertura_texto:.1f}%")
    print(f"Aspectos identificados: {aspectos_identificados}")
    print(f"Calidad general: {calidad_general:.1f}%")
    
    # paso 7: visualizaciones
    print(f"\n--- Generando visualizaciones ---")
    generar_visualizaciones_dataset(df_final, aspectos_stats, nombre_dataset)
    
    return {
        'nombre_dataset': nombre_dataset,
        'fuente': info_dataset['descripcion'],
        'dataset_final': df_final,
        'aspectos_detectados': aspectos_stats,
        'pain_points': pain_points,
        'recomendaciones': recomendaciones,
        'metricas_calidad': {
            'total_registros': len(df),
            'registros_procesados': len(df_final),
            'cobertura_texto': cobertura_texto,
            'confianza_promedio': confianza_promedio,
            'nps_estimado': nps,
            'aspectos_identificados': aspectos_identificados,
            'calidad_general': calidad_general
        },
        'estadisticas_sentimiento': {
            'total_opiniones': total,
            'promotores': promotores,
            'detractores': detractores,
            'neutrales': distribucion.get('neutral', 0),
            'distribucion': distribucion.to_dict()
        }
    }

def generar_visualizaciones_dataset(df, aspectos_stats, nombre_dataset):
    """
    genera visualizaciones específicas para cada dataset
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Análisis de opiniones - {nombre_dataset}', fontsize=16, fontweight='bold')
        
        # gráfico 1: distribución de sentimientos
        if 'sentimiento_vader' in df.columns:
            sentiment_counts = df['sentimiento_vader'].value_counts()
            colores = ['#27ae60', '#f39c12', '#e74c3c']  # verde, amarillo, rojo
            
            if len(sentiment_counts) > 0:
                axes[0,0].pie(sentiment_counts.values, 
                             labels=[s.capitalize() for s in sentiment_counts.index],
                             autopct='%1.1f%%', 
                             colors=colores[:len(sentiment_counts)],
                             startangle=90)
                axes[0,0].set_title('Distribución de sentimientos')
        
        # gráfico 2: aspectos mencionados
        aspectos_totales = {k: v['menciones_totales'] for k, v in aspectos_stats.items() 
                           if v['menciones_totales'] > 0}
        
        if aspectos_totales:
            aspectos_nombres = list(aspectos_totales.keys())[:6]  # top 6
            aspectos_valores = [aspectos_totales[asp] for asp in aspectos_nombres]
            
            bars = axes[0,1].bar(aspectos_nombres, aspectos_valores, color='#3498db', alpha=0.8)
            axes[0,1].set_title('Aspectos más mencionados')
            axes[0,1].set_xlabel('Aspectos')
            axes[0,1].set_ylabel('Menciones')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # valores en barras
            for bar, valor in zip(bars, aspectos_valores):
                height = bar.get_height()
                axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                              f'{valor}', ha='center', va='bottom', fontsize=9)
        
        # gráfico 3: aspectos positivos vs negativos
        aspectos_para_grafico = []
        positivos_vals = []
        negativos_vals = []
        
        for aspecto, stats in aspectos_stats.items():
            if stats['menciones_totales'] >= 2:
                aspectos_para_grafico.append(aspecto[:8])
                positivos_vals.append(stats['menciones_positivas'])
                negativos_vals.append(stats['menciones_negativas'])
        
        if aspectos_para_grafico:
            x_pos = np.arange(len(aspectos_para_grafico))
            axes[1,0].bar(x_pos, positivos_vals, label='Positivas', color='#27ae60', alpha=0.8)
            axes[1,0].bar(x_pos, negativos_vals, bottom=positivos_vals, label='Negativas', 
                         color='#e74c3c', alpha=0.8)
            
            axes[1,0].set_title('Menciones positivas vs negativas por aspecto')
            axes[1,0].set_xlabel('Aspectos')
            axes[1,0].set_ylabel('Número de menciones')
            axes[1,0].set_xticks(x_pos)
            axes[1,0].set_xticklabels(aspectos_para_grafico, rotation=45)
            axes[1,0].legend()
        
        # gráfico 4: distribución de confianza
        if 'confianza' in df.columns:
            confianza_data = df['confianza'].dropna()
            if len(confianza_data) > 0:
                axes[1,1].hist(confianza_data, bins=15, alpha=0.7, color='#9b59b6', edgecolor='black')
                axes[1,1].set_title('Distribución de confianza del análisis')
                axes[1,1].set_xlabel('Nivel de confianza')
                axes[1,1].set_ylabel('Frecuencia')
                
                promedio = confianza_data.mean()
                axes[1,1].axvline(x=promedio, color='red', linestyle='--', 
                                 label=f'Promedio: {promedio:.2f}')
                axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(f'analisis_{nombre_dataset.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # generar wordcloud si hay suficientes datos
        if len(df) > 10 and 'tokens' in df.columns:
            generar_wordcloud_dataset(df, nombre_dataset)
            
    except Exception as e:
        print(f"Error generando visualizaciones para {nombre_dataset}: {e}")

def generar_wordcloud_dataset(df, nombre_dataset):
    """
    genera wordcloud específico para el dataset
    """
    try:
        todos_tokens = []
        for tokens_list in df['tokens'].dropna():
            if isinstance(tokens_list, list):
                todos_tokens.extend(tokens_list)
        
        if len(todos_tokens) > 10:
            frecuencias = Counter(todos_tokens)
            texto_wordcloud = ' '.join([palabra for palabra, freq in frecuencias.most_common(50)])
            
            if texto_wordcloud.strip():
                plt.figure(figsize=(12, 6))
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    max_words=50,
                                    colormap='viridis').generate(texto_wordcloud)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Términos más frecuentes - {nombre_dataset}', fontweight='bold', fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'wordcloud_{nombre_dataset.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
    except Exception as e:
        print(f"Error generando wordcloud para {nombre_dataset}: {e}")

# ===============================
# CONSOLIDACIÓN DE RESULTADOS
# ===============================

def generar_informe_consolidado(resultados_por_dataset):
    """
    genera informe consolidado combinando todos los datasets analizados
    """
    print(f"\n{'='*85}")
    print("                    INFORME CONSOLIDADO FINAL")
    print("            ANÁLISIS DE OPINIONES POR DATASET")
    print("                Samsung Innovation Campus 2024")
    print(f"{'='*85}")
    
    if not resultados_por_dataset:
        print("No hay resultados para consolidar")
        return
    
    # estadísticas generales
    total_datasets = len(resultados_por_dataset)
    total_registros = sum(r['metricas_calidad']['total_registros'] for r in resultados_por_dataset)
    total_procesados = sum(r['metricas_calidad']['registros_procesados'] for r in resultados_por_dataset)
    
    print(f"\nRESUMEN GENERAL:")
    print("-" * 50)
    print(f"• Datasets analizados: {total_datasets}")
    print(f"• Total de registros originales: {total_registros:,}")
    print(f"• Total de registros procesados: {total_procesados:,}")
    print(f"• Cobertura general: {(total_procesados/total_registros*100):.1f}%")
    
    # análisis por dataset
    print(f"\nANÁLISIS POR DATASET:")
    print("-" * 50)
    
    for i, resultado in enumerate(resultados_por_dataset, 1):
        nombre = resultado['nombre_dataset']
        metricas = resultado['metricas_calidad']
        sentimientos = resultado['estadisticas_sentimiento']
        
        print(f"\n{i}. DATASET: {nombre.upper()}")
        print(f"   Fuente: {resultado['fuente']}")
        print(f"   Registros procesados: {metricas['registros_procesados']:,}")
        print(f"   NPS estimado: {metricas['nps_estimado']:.1f}")
        print(f"   Calidad del análisis: {metricas['calidad_general']:.1f}%")
        print(f"   Aspectos identificados: {metricas['aspectos_identificados']}")
        print(f"   Pain points detectados: {len(resultado['pain_points'])}")
        
        # distribución de sentimientos
        total_sent = sentimientos['total_opiniones']
        if total_sent > 0:
            print(f"   Sentimientos: {sentimientos['promotores']} pos, "
                  f"{sentimientos['neutrales']} neu, {sentimientos['detractores']} neg")
    
    # consolidación de aspectos críticos
    print(f"\nASPECTOS CRÍTICOS CONSOLIDADOS:")
    print("-" * 50)
    
    aspectos_consolidados = {}
    
    for resultado in resultados_por_dataset:
        for pain_point in resultado['pain_points']:
            aspecto = pain_point['aspecto']
            if aspecto not in aspectos_consolidados:
                aspectos_consolidados[aspecto] = {
                    'datasets_afectados': [],
                    'total_menciones_negativas': 0,
                    'severidad_promedio': 0,
                    'datasets_count': 0
                }
            
            aspectos_consolidados[aspecto]['datasets_afectados'].append(resultado['nombre_dataset'])
            aspectos_consolidados[aspecto]['total_menciones_negativas'] += pain_point['menciones_negativas']
            aspectos_consolidados[aspecto]['severidad_promedio'] += pain_point['severidad']
            aspectos_consolidados[aspecto]['datasets_count'] += 1
    
    # calcular promedios y ordenar
    for aspecto, datos in aspectos_consolidados.items():
        datos['severidad_promedio'] = datos['severidad_promedio'] / datos['datasets_count']
    
    aspectos_ordenados = sorted(aspectos_consolidados.items(), 
                               key=lambda x: x[1]['severidad_promedio'], reverse=True)
    
    for i, (aspecto, datos) in enumerate(aspectos_ordenados[:5], 1):
        print(f"{i}. {aspecto.upper()}:")
        print(f"   Datasets afectados: {datos['datasets_count']}/{total_datasets}")
        print(f"   Total menciones negativas: {datos['total_menciones_negativas']}")
        print(f"   Severidad promedio: {datos['severidad_promedio']:.2f}")
        print(f"   Datasets: {', '.join(datos['datasets_afectados'])}")
    
    # recomendaciones consolidadas
    print(f"\nRECOMENDACIONES ESTRATÉGICAS CONSOLIDADAS:")
    print("-" * 50)
    
    recomendaciones_consolidadas = generar_recomendaciones_consolidadas(aspectos_ordenados, resultados_por_dataset)
    
    for i, rec in enumerate(recomendaciones_consolidadas[:5], 1):
        print(f"\n{i}. RECOMENDACIÓN PRIORITARIA - {rec['aspecto'].upper()}")
        print(f"   Estrategia: {rec['recomendacion']}")
        print(f"   Impacto esperado: {rec['impacto_esperado']}")
        print(f"   Aplicable a: {rec['datasets_aplicables']} datasets")
        print("   Acciones específicas:")
        for accion in rec['acciones']:
            print(f"     • {accion}")
    
    # métricas comparativas
    print(f"\nMÉTRICAS COMPARATIVAS:")
    print("-" * 50)
    
    nps_valores = [r['metricas_calidad']['nps_estimado'] for r in resultados_por_dataset]
    calidad_valores = [r['metricas_calidad']['calidad_general'] for r in resultados_por_dataset]
    
    if nps_valores:
        print(f"NPS promedio general: {np.mean(nps_valores):.1f}")
        print(f"NPS rango: {min(nps_valores):.1f} a {max(nps_valores):.1f}")
        print(f"Calidad promedio: {np.mean(calidad_valores):.1f}%")
        
        # identificar mejor y peor dataset
        mejor_dataset_idx = nps_valores.index(max(nps_valores))
        peor_dataset_idx = nps_valores.index(min(nps_valores))
        
        print(f"Mejor performance: {resultados_por_dataset[mejor_dataset_idx]['nombre_dataset']} "
              f"(NPS: {max(nps_valores):.1f})")
        print(f"Requiere atención: {resultados_por_dataset[peor_dataset_idx]['nombre_dataset']} "
              f"(NPS: {min(nps_valores):.1f})")
    
    # plan de acción integrado
    print(f"\nPLAN DE ACCIÓN INTEGRADO:")
    print("-" * 50)
    print("FASE 1 - Atención inmediata (0-1 mes):")
    print("  • Abordar aspectos críticos presentes en múltiples datasets")
    print("  • Implementar mejoras rápidas en datasets con NPS negativo")
    print("  • Establecer sistema de monitoreo unificado")
    
    print("\nFASE 2 - Mejoras estructurales (1-3 meses):")
    print("  • Implementar recomendaciones específicas por dataset")
    print("  • Desarrollar estrategias diferenciadas por fuente de datos")
    print("  • Crear dashboard consolidado de seguimiento")
    
    print("\nFASE 3 - Optimización continua (3-6 meses):")
    print("  • Evaluar impacto de mejoras implementadas")
    print("  • Expandir mejores prácticas entre datasets")
    print("  • Automatizar pipeline de análisis continuo")
    
    return {
        'datasets_analizados': total_datasets,
        'aspectos_criticos_consolidados': aspectos_consolidados,
        'recomendaciones_consolidadas': recomendaciones_consolidadas,
        'metricas_generales': {
            'nps_promedio': np.mean(nps_valores) if nps_valores else 0,
            'calidad_promedio': np.mean(calidad_valores) if calidad_valores else 0,
            'total_registros': total_registros,
            'total_procesados': total_procesados
        }
    }

def generar_recomendaciones_consolidadas(aspectos_ordenados, resultados_por_dataset):
    """
    genera recomendaciones estratégicas basadas en análisis consolidado
    """
    plantillas_consolidadas = {
        'calidad': {
            'recomendacion': 'Implementar programa integral de mejora de calidad',
            'acciones': [
                'Establecer estándares de calidad unificados entre todas las fuentes',
                'Implementar sistema de control de calidad en tiempo real',
                'Capacitar equipos en mejores prácticas de calidad',
                'Crear métricas de calidad comparables entre datasets'
            ],
            'impacto_esperado': 'Alto - Mejora directa en satisfacción'
        },
        'precio': {
            'recomendacion': 'Revisar estrategia de precios y comunicación de valor',
            'acciones': [
                'Análisis competitivo de precios por segmento',
                'Mejorar comunicación de propuesta de valor',
                'Implementar pricing dinámico basado en feedback',
                'Desarrollar programas de valor agregado'
            ],
            'impacto_esperado': 'Medio-Alto - Mejora en percepción de valor'
        },
        'servicio': {
            'recomendacion': 'Transformar experiencia de servicio al cliente',
            'acciones': [
                'Unificar estándares de servicio entre canales',
                'Implementar CRM integrado con análisis de sentimiento',
                'Capacitación avanzada en servicio al cliente',
                'Sistema de escalación automática para casos críticos'
            ],
            'impacto_esperado': 'Muy Alto - Impacto directo en NPS'
        },
        'producto': {
            'recomendacion': 'Optimizar portfolio de productos basado en feedback',
            'acciones': [
                'Priorizar desarrollo basado en análisis de aspectos',
                'Implementar feedback loop continuo con desarrollo',
                'Testing A/B de nuevas características',
                'Roadmap de producto dirigido por datos de satisfacción'
            ],
            'impacto_esperado': 'Alto - Mejora en satisfacción a largo plazo'
        },
        'experiencia': {
            'recomendacion': 'Rediseñar experiencia de usuario integral',
            'acciones': [
                'Mapear customer journey completo',
                'Simplificar procesos identificados como problemáticos',
                'Implementar UX/UI centrado en feedback de usuarios',
                'Sistema de feedback en tiempo real en touchpoints clave'
            ],
            'impacto_esperado': 'Muy Alto - Mejora en experiencia general'
        },
        'entrega': {
            'recomendacion': 'Optimizar cadena de entrega y comunicación',
            'acciones': [
                'Implementar tracking en tiempo real',
                'Mejorar comunicación proactiva de estados',
                'Optimizar tiempos de entrega por región',
                'Sistema de alertas automáticas para delays'
            ],
            'impacto_esperado': 'Alto - Mejora en satisfacción operacional'
        },
        'diseño': {
            'recomendacion': 'Mejorar aspectos de diseño y presentación',
            'acciones': [
                'Revisión de guidelines de diseño',
                'Testing de usabilidad basado en feedback',
                'Implementar design system consistente',
                'Validación continua con usuarios finales'
            ],
            'impacto_esperado': 'Medio - Mejora en percepción de marca'
        }
    }
    
    recomendaciones = []
    
    for aspecto, datos in aspectos_ordenados[:5]:
        template = plantillas_consolidadas.get(aspecto, {
            'recomendacion': f'Desarrollar estrategia específica para {aspecto}',
            'acciones': [
                f'Análisis profundo de problemas en {aspecto}',
                f'Implementar mejoras basadas en feedback',
                f'Monitorear progreso en {aspecto}',
                f'Escalar mejores prácticas'
            ],
            'impacto_esperado': 'Medio - Basado en análisis específico'
        })
        
        recomendaciones.append({
            'aspecto': aspecto,
            'recomendacion': template['recomendacion'],
            'acciones': template['acciones'],
            'impacto_esperado': template['impacto_esperado'],
            'datasets_aplicables': datos['datasets_count'],
            'prioridad': 'ALTA' if datos['severidad_promedio'] > 1.0 else 'MEDIA',
            'datasets_afectados': datos['datasets_afectados']
        })
    
    return recomendaciones

# ===============================
# PIPELINE PRINCIPAL MEJORADO
# ===============================

def ejecutar_pipeline_completo_multi_dataset():
    """
    pipeline principal que procesa todos los datasets individualmente
    """
    print("SISTEMA INTELIGENTE DE ANÁLISIS DE OPINIONES")
    print("Metodología Samsung Innovation Campus - Análisis por Dataset")
    print("="*75)
    
    try:
        # paso 1: cargar todos los datasets
        datasets = cargar_todos_los_datasets()
        
        if not datasets:
            print("Error: No se encontraron datasets válidos")
            return None
        
        # paso 2: procesar cada dataset individualmente
        resultados_por_dataset = []
        
        for nombre_dataset, info_dataset in datasets.items():
            try:
                resultado = procesar_dataset_individual(nombre_dataset, info_dataset)
                if resultado:
                    resultados_por_dataset.append(resultado)
                    print(f"\n✓ Dataset {nombre_dataset} procesado exitosamente")
                else:
                    print(f"\n✗ Error procesando dataset {nombre_dataset}")
            except Exception as e:
                print(f"\n✗ Error en dataset {nombre_dataset}: {e}")
                continue
        
        if not resultados_por_dataset:
            print("Error: No se pudo procesar ningún dataset")
            return None
        
        # paso 3: consolidar resultados
        print(f"\n{'='*75}")
        print("CONSOLIDANDO RESULTADOS DE TODOS LOS DATASETS")
        print(f"{'='*75}")
        
        informe_consolidado = generar_informe_consolidado(resultados_por_dataset)
        
        # resumen final
        print(f"\n{'='*75}")
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print(f"{'='*75}")
        print(f"Datasets procesados: {len(resultados_por_dataset)}/{len(datasets)}")
        
        total_registros = sum(r['metricas_calidad']['registros_procesados'] for r in resultados_por_dataset)
        print(f"Total registros analizados: {total_registros:,}")
        
        total_aspectos = sum(r['metricas_calidad']['aspectos_identificados'] for r in resultados_por_dataset)
        print(f"Total aspectos identificados: {total_aspectos}")
        
        total_pain_points = sum(len(r['pain_points']) for r in resultados_por_dataset)
        print(f"Total pain points detectados: {total_pain_points}")
        
        total_recomendaciones = len(informe_consolidado['recomendaciones_consolidadas'])
        print(f"Recomendaciones consolidadas: {total_recomendaciones}")
        
        if informe_consolidado['metricas_generales']['nps_promedio']:
            print(f"NPS promedio general: {informe_consolidado['metricas_generales']['nps_promedio']:.1f}")
        
        print(f"\nVisualizaciones generadas para cada dataset")
        print(f"Informe consolidado completado")
        print("="*75)
        
        return {
            'resultados_individuales': resultados_por_dataset,
            'informe_consolidado': informe_consolidado,
            'metricas_generales': informe_consolidado['metricas_generales'],
            'datasets_procesados': len(resultados_por_dataset)
        }
        
    except Exception as e:
        print(f"\nError crítico en pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===============================
# EJECUCIÓN PRINCIPAL
# ===============================

if __name__ == "__main__":
    try:
        print("INICIANDO ANÁLISIS MULTI-DATASET...")
        print("Aplicando metodología completa a cada dataset individual")
        print()
        
        # ejecutar pipeline completo
        resultados_finales = ejecutar_pipeline_completo_multi_dataset()
        
        if resultados_finales:
            print("\n🎯 SISTEMA EJECUTADO EXITOSAMENTE")
            print("✓ Análisis individual completado para cada dataset")
            print("✓ Informe consolidado generado")
            print("✓ Visualizaciones creadas")
            print("✓ Recomendaciones estratégicas disponibles")
            
            # mostrar resumen de datasets procesados
            print(f"\nDATASETS PROCESADOS:")
            for resultado in resultados_finales['resultados_individuales']:
                nombre = resultado['nombre_dataset']
                nps = resultado['metricas_calidad']['nps_estimado']
                registros = resultado['metricas_calidad']['registros_procesados']
                print(f"  • {nombre}: {registros:,} registros, NPS: {nps:.1f}")
            
        else:
            print("\nERROR: No se pudieron procesar los datos")
            print("Verificar:")
            print("- Que los archivos CSV estén en el directorio actual")
            print("- Que el archivo de Amazon tenga el nombre correcto")
            print("- Que los archivos tengan contenido válido")
        
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario")
    except Exception as e:
        print(f"\nError crítico: {e}")
        import traceback
        traceback.print_exc()