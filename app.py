from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from modelo.preprocesamiento import cargar_y_preprocesar
from modelo.mlp_model import detectar_anomalias, mapear_etiqueta
from modelo.reporte import generar_reporte

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GRAFICAS_FOLDER'] = 'static/graficas'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analizar', methods=['POST'])
def analizar():
    if 'archivo' not in request.files:
        return redirect(url_for('index'))

    archivo = request.files['archivo']
    if archivo.filename == '':
        return redirect(url_for('index'))

    # Guardar CSV
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
    archivo.save(filepath)

    # Preprocesar
    X, columnas, le, scaler = cargar_y_preprocesar(filepath)

    # Detectar anomalías con MLP
    predicciones, probabilidades, kmeans, mlp = detectar_anomalias(X)

    # Cargar CSV original para mostrar resultados
    df_original = pd.read_csv(filepath)
    df_original.columns = df_original.columns.str.strip().str.replace('"', '')

    # Agregar clasificación
    df_original = df_original.iloc[:len(predicciones)].copy()
    df_original['Clasificacion'] = [mapear_etiqueta(p) for p in predicciones]
    df_original['Confianza'] = [f"{max(prob)*100:.1f}%" for prob in probabilidades]

    # Generar gráficas
    graficas = []

    # Gráfica 1: Distribución de clasificaciones
    fig, ax = plt.subplots(figsize=(7, 4))
    conteos = df_original['Clasificacion'].value_counts()
    colores = {'Normal': '#2ecc71', 'Sospechoso': '#f39c12', 'Posible Ataque': '#e74c3c'}
    conteos.plot(kind='bar', ax=ax,
                 color=[colores.get(c, 'grey') for c in conteos.index])
    ax.set_title('Distribución de Clasificaciones')
    ax.set_xlabel('Clasificación')
    ax.set_ylabel('Cantidad de Paquetes')
    plt.tight_layout()
    g1 = os.path.join(app.config['GRAFICAS_FOLDER'], 'distribucion.png')
    plt.savefig(g1)
    plt.close()
    graficas.append(g1)

    # Gráfica 2: Protocolos más usados
    fig, ax = plt.subplots(figsize=(7, 4))
    df_original['Protocol'].value_counts().head(10).plot(kind='bar', ax=ax, color='#3498db')
    ax.set_title('Top 10 Protocolos')
    ax.set_xlabel('Protocolo')
    ax.set_ylabel('Cantidad')
    plt.tight_layout()
    g2 = os.path.join(app.config['GRAFICAS_FOLDER'], 'protocolos.png')
    plt.savefig(g2)
    plt.close()
    graficas.append(g2)

    # Gráfica 3: Longitud de paquetes por clasificación
    fig, ax = plt.subplots(figsize=(7, 4))
    df_original['Length'] = pd.to_numeric(df_original['Length'], errors='coerce')
    for clasificacion, color in colores.items():
        subset = df_original[df_original['Clasificacion'] == clasificacion]['Length'].dropna()
        if not subset.empty:
            ax.hist(subset, bins=30, alpha=0.6, label=clasificacion, color=color)
    ax.set_title('Distribución de Longitud de Paquetes')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    plt.tight_layout()
    g3 = os.path.join(app.config['GRAFICAS_FOLDER'], 'longitud.png')
    plt.savefig(g3)
    plt.close()
    graficas.append(g3)

    # Resumen para el template
    resumen = {
        'total': len(df_original),
        'normales': len(df_original[df_original['Clasificacion'] == 'Normal']),
        'sospechosos': len(df_original[df_original['Clasificacion'] == 'Sospechoso']),
        'ataques': len(df_original[df_original['Clasificacion'] == 'Posible Ataque']),
    }

    # Guardar para reporte PDF
    df_original.to_csv('uploads/resultado_analisis.csv', index=False)

    # Convertir tabla a HTML
    tabla_html = df_original.head(100).to_html(
        classes='table table-striped table-bordered',
        index=False
    )

    return render_template('resultados.html',
                           tabla=tabla_html,
                           resumen=resumen,
                           graficas=[
                               'graficas/distribucion.png',
                               'graficas/protocolos.png',
                               'graficas/longitud.png'
                           ])

@app.route('/descargar_reporte')
def descargar_reporte():
    df = pd.read_csv('uploads/resultado_analisis.csv')
    graficas = [
        'static/graficas/distribucion.png',
        'static/graficas/protocolos.png',
        'static/graficas/longitud.png'
    ]
    reporte_path = 'uploads/reporte_auditoria.pdf'
    generar_reporte(df, graficas, reporte_path)
    return send_file(reporte_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)