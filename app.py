from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from modelo.preprocesamiento import cargar_y_preprocesar
from modelo.mlp_model import detectar_anomalias, mapear_etiqueta, obtener_metricas
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

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
    archivo.save(filepath)

    X, columnas, le, scaler = cargar_y_preprocesar(filepath)
    resultados = detectar_anomalias(X)
    metricas = obtener_metricas(
        resultados['etiquetas'],
        resultados['pred_mlp'],
        resultados['pred_rf'],
        resultados['prob_mlp'],
        resultados['prob_rf']
    )

    df_original = pd.read_csv(filepath)
    df_original.columns = df_original.columns.str.strip().str.replace('"', '')
    df_original = df_original.iloc[:len(resultados['pred_mlp'])].copy()
    df_original['Clasificacion_MLP'] = [mapear_etiqueta(p) for p in resultados['pred_mlp']]
    df_original['Clasificacion_RF'] = [mapear_etiqueta(p) for p in resultados['pred_rf']]
    df_original['Confianza_MLP'] = [f"{max(p)*100:.1f}%" for p in resultados['prob_mlp']]
    df_original['Confianza_RF'] = [f"{max(p)*100:.1f}%" for p in resultados['prob_rf']]

    colores = {'Normal': '#2ecc71', 'Sospechoso': '#f39c12', 'Posible Ataque': '#e74c3c'}
    graficas = []

    # Gráfica 1: Distribución clasificaciones MLP
    fig, ax = plt.subplots(figsize=(7, 4))
    conteos = df_original['Clasificacion_MLP'].value_counts()
    conteos.plot(kind='bar', ax=ax,
                 color=[colores.get(c, 'grey') for c in conteos.index])
    ax.set_title('Distribución de Clasificaciones (MLP)')
    ax.set_xlabel('Clasificación')
    ax.set_ylabel('Cantidad')
    plt.tight_layout()
    g1 = os.path.join(app.config['GRAFICAS_FOLDER'], 'distribucion.png')
    plt.savefig(g1); plt.close()
    graficas.append('graficas/distribucion.png')

    # Gráfica 2: Top protocolos
    fig, ax = plt.subplots(figsize=(7, 4))
    df_original['Protocol'].value_counts().head(10).plot(kind='bar', ax=ax, color='#3498db')
    ax.set_title('Top 10 Protocolos')
    ax.set_xlabel('Protocolo')
    ax.set_ylabel('Cantidad')
    plt.tight_layout()
    g2 = os.path.join(app.config['GRAFICAS_FOLDER'], 'protocolos.png')
    plt.savefig(g2); plt.close()
    graficas.append('graficas/protocolos.png')

    # Gráfica 3: Longitud de paquetes
    fig, ax = plt.subplots(figsize=(7, 4))
    df_original['Length'] = pd.to_numeric(df_original['Length'], errors='coerce')
    for clf, color in colores.items():
        subset = df_original[df_original['Clasificacion_MLP'] == clf]['Length'].dropna()
        if not subset.empty:
            ax.hist(subset, bins=30, alpha=0.6, label=clf, color=color)
    ax.set_title('Distribución de Longitud de Paquetes')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    plt.tight_layout()
    g3 = os.path.join(app.config['GRAFICAS_FOLDER'], 'longitud.png')
    plt.savefig(g3); plt.close()
    graficas.append('graficas/longitud.png')

    # Gráfica 4: Clusters PCA
    fig, ax = plt.subplots(figsize=(7, 5))
    X_pca = resultados['X_pca']
    etiquetas = resultados['etiquetas']
    scatter_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels_map = {0: 'Normal', 1: 'Sospechoso', 2: 'Posible Ataque'}
    for i in range(3):
        mask = etiquetas == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=scatter_colors[i], label=labels_map[i],
                   alpha=0.5, s=10)
    ax.set_title('Visualización de Clusters (PCA 2D)')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.legend()
    plt.tight_layout()
    g4 = os.path.join(app.config['GRAFICAS_FOLDER'], 'clusters_pca.png')
    plt.savefig(g4); plt.close()
    graficas.append('graficas/clusters_pca.png')

    # Gráfica 5: Matriz de confusión MLP
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    clases = ['Normal', 'Sospechoso', 'Posible Ataque']
    for ax, cm, titulo in zip(axes,
                               [metricas['cm_mlp'], metricas['cm_rf']],
                               ['Matriz de Confusión - MLP', 'Matriz de Confusión - RF']):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=clases, yticklabels=clases, ax=ax)
        ax.set_title(titulo)
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
    plt.tight_layout()
    g5 = os.path.join(app.config['GRAFICAS_FOLDER'], 'confusion.png')
    plt.savefig(g5); plt.close()
    graficas.append('graficas/confusion.png')

    # Gráfica 6: Curvas ROC
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    roc_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for ax, roc_data, titulo in zip(axes,
                                     [metricas['roc_mlp'], metricas['roc_rf']],
                                     ['Curvas ROC - MLP', 'Curvas ROC - RF']):
        for i, (clase, color) in enumerate(zip(clases, roc_colors)):
            ax.plot(roc_data[clase]['fpr'], roc_data[clase]['tpr'],
                    color=color,
                    label=f"{clase} (AUC={roc_data[clase]['auc']})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_title(titulo)
        ax.set_xlabel('Tasa Falsos Positivos')
        ax.set_ylabel('Tasa Verdaderos Positivos')
        ax.legend(fontsize=8)
    plt.tight_layout()
    g6 = os.path.join(app.config['GRAFICAS_FOLDER'], 'roc.png')
    plt.savefig(g6); plt.close()
    graficas.append('graficas/roc.png')

    resumen = {
        'total': len(df_original),
        'normales': len(df_original[df_original['Clasificacion_MLP'] == 'Normal']),
        'sospechosos': len(df_original[df_original['Clasificacion_MLP'] == 'Sospechoso']),
        'ataques': len(df_original[df_original['Clasificacion_MLP'] == 'Posible Ataque']),
    }

    # Métricas para template
    metricas_template = {
        'mlp': {
            'accuracy': round(metricas['reporte_mlp']['accuracy'] * 100, 1),
            'precision': round(metricas['reporte_mlp']['weighted avg']['precision'] * 100, 1),
            'recall': round(metricas['reporte_mlp']['weighted avg']['recall'] * 100, 1),
            'f1': round(metricas['reporte_mlp']['weighted avg']['f1-score'] * 100, 1),
        },
        'rf': {
            'accuracy': round(metricas['reporte_rf']['accuracy'] * 100, 1),
            'precision': round(metricas['reporte_rf']['weighted avg']['precision'] * 100, 1),
            'recall': round(metricas['reporte_rf']['weighted avg']['recall'] * 100, 1),
            'f1': round(metricas['reporte_rf']['weighted avg']['f1-score'] * 100, 1),
        }
    }

    df_original.to_csv('uploads/resultado_analisis.csv', index=False)

    tabla_html = df_original.head(100).to_html(
        classes='table table-striped table-bordered',
        index=False
    )

    return render_template('resultados.html',
                           tabla=tabla_html,
                           resumen=resumen,
                           graficas=graficas,
                           metricas=metricas_template)

@app.route('/descargar_reporte')
def descargar_reporte():
    df = pd.read_csv('uploads/resultado_analisis.csv')
    graficas = [
        'static/graficas/distribucion.png',
        'static/graficas/protocolos.png',
        'static/graficas/longitud.png',
        'static/graficas/clusters_pca.png',
        'static/graficas/confusion.png',
        'static/graficas/roc.png'
    ]
    reporte_path = 'uploads/reporte_auditoria.pdf'
    generar_reporte(df, graficas, reporte_path)
    return send_file(reporte_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)