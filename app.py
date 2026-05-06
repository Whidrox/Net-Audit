from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import uuid
import threading
import glob
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

os.makedirs('uploads', exist_ok=True)
os.makedirs('static/graficas', exist_ok=True)

tareas = {}

def limpiar_archivos_viejos():
    carpetas = ['static/graficas', 'uploads']
    for carpeta in carpetas:
        archivos = glob.glob(f'{carpeta}/*')
        archivos_con_id = [a for a in archivos if '_' in os.path.basename(a)
                          and os.path.basename(a).count('_') >= 2]
        archivos_con_id.sort(key=lambda x: os.path.getmtime(x))
        if len(archivos_con_id) > 5:
            for archivo in archivos_con_id[:-5]:
                try:
                    os.remove(archivo)
                except:
                    pass

def procesar_archivo(task_id, filepath):
    try:
        tareas[task_id]['estado'] = 'procesando'

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
        carpeta = app.config['GRAFICAS_FOLDER']

        # Gráfica 1: Distribución
        fig, ax = plt.subplots(figsize=(7, 4))
        conteos = df_original['Clasificacion_MLP'].value_counts()
        conteos.plot(kind='bar', ax=ax,
                     color=[colores.get(c, 'grey') for c in conteos.index])
        ax.set_title('Distribución de Clasificaciones (MLP)')
        ax.set_xlabel('Clasificación')
        ax.set_ylabel('Cantidad')
        plt.tight_layout()
        plt.savefig(f'{carpeta}/distribucion_{task_id}.png')
        plt.close()

        # Gráfica 2: Protocolos
        fig, ax = plt.subplots(figsize=(7, 4))
        df_original['Protocol'].value_counts().head(10).plot(kind='bar', ax=ax, color='#3498db')
        ax.set_title('Top 10 Protocolos')
        plt.tight_layout()
        plt.savefig(f'{carpeta}/protocolos_{task_id}.png')
        plt.close()

        # Gráfica 3: Longitud
        fig, ax = plt.subplots(figsize=(7, 4))
        df_original['Length'] = pd.to_numeric(df_original['Length'], errors='coerce')
        for clf, color in colores.items():
            subset = df_original[df_original['Clasificacion_MLP'] == clf]['Length'].dropna()
            if not subset.empty:
                ax.hist(subset, bins=30, alpha=0.6, label=clf, color=color)
        ax.set_title('Distribución de Longitud de Paquetes')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{carpeta}/longitud_{task_id}.png')
        plt.close()

        # Gráfica 4: Clusters PCA
        fig, ax = plt.subplots(figsize=(7, 5))
        X_pca = resultados['X_pca']
        etiquetas = resultados['etiquetas']
        scatter_colors = ['#2ecc71', '#f39c12', '#e74c3c']
        labels_map = {0: 'Normal', 1: 'Sospechoso', 2: 'Posible Ataque'}
        for i in range(3):
            mask = etiquetas == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=scatter_colors[i], label=labels_map[i], alpha=0.5, s=10)
        ax.set_title('Visualización de Clusters (PCA 2D)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{carpeta}/clusters_pca_{task_id}.png')
        plt.close()

        # Gráfica 5: Matrices de confusión
        clases = ['Normal', 'Sospechoso', 'Posible Ataque']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, cm, titulo in zip(axes,
                                   [metricas['cm_mlp'], metricas['cm_rf']],
                                   ['Matriz de Confusión - MLP', 'Matriz de Confusión - RF']):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=clases, yticklabels=clases, ax=ax)
            ax.set_title(titulo)
            ax.set_ylabel('Real')
            ax.set_xlabel('Predicho')
        plt.tight_layout()
        plt.savefig(f'{carpeta}/confusion_{task_id}.png')
        plt.close()

        # Gráfica 6: ROC
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        roc_colors = ['#2ecc71', '#f39c12', '#e74c3c']
        for ax, roc_data, titulo in zip(axes,
                                         [metricas['roc_mlp'], metricas['roc_rf']],
                                         ['Curvas ROC - MLP', 'Curvas ROC - RF']):
            for clase, color in zip(clases, roc_colors):
                ax.plot(roc_data[clase]['fpr'], roc_data[clase]['tpr'],
                        color=color,
                        label=f"{clase} (AUC={roc_data[clase]['auc']})")
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_title(titulo)
            ax.set_xlabel('Falsos Positivos')
            ax.set_ylabel('Verdaderos Positivos')
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{carpeta}/roc_{task_id}.png')
        plt.close()

        resumen = {
            'total': len(df_original),
            'normales': len(df_original[df_original['Clasificacion_MLP'] == 'Normal']),
            'sospechosos': len(df_original[df_original['Clasificacion_MLP'] == 'Sospechoso']),
            'ataques': len(df_original[df_original['Clasificacion_MLP'] == 'Posible Ataque']),
        }

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

        resultado_csv = f'uploads/resultado_{task_id}.csv'
        df_original.to_csv(resultado_csv, index=False)

        tareas[task_id] = {
            'estado': 'listo',
            'resumen': resumen,
            'metricas': metricas_template,
            'task_id': task_id,
            'tabla': df_original.head(100).to_html(
                classes='table table-striped table-bordered',
                index=False
            )
        }

    except Exception as e:
        tareas[task_id] = {'estado': 'error', 'mensaje': str(e)}


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

    task_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{task_id}_{archivo.filename}')
    archivo.save(filepath)

    limpiar_archivos_viejos()

    tareas[task_id] = {'estado': 'iniciando'}
    hilo = threading.Thread(target=procesar_archivo, args=(task_id, filepath))
    hilo.start()

    return render_template('cargando.html', task_id=task_id)


@app.route('/estado/<task_id>')
def estado(task_id):
    tarea = tareas.get(task_id, {'estado': 'no encontrado'})
    return jsonify({'estado': tarea['estado']})


@app.route('/resultados/<task_id>')
def resultados(task_id):
    tarea = tareas.get(task_id)
    if not tarea or tarea['estado'] != 'listo':
        return redirect(url_for('index'))
    return render_template('resultados.html',
                           tabla=tarea['tabla'],
                           resumen=tarea['resumen'],
                           metricas=tarea['metricas'],
                           task_id=task_id)


@app.route('/descargar_reporte/<task_id>')
def descargar_reporte(task_id):
    df = pd.read_csv(f'uploads/resultado_{task_id}.csv')
    carpeta = 'static/graficas'
    graficas = [
        f'{carpeta}/distribucion_{task_id}.png',
        f'{carpeta}/protocolos_{task_id}.png',
        f'{carpeta}/longitud_{task_id}.png',
        f'{carpeta}/clusters_pca_{task_id}.png',
        f'{carpeta}/confusion_{task_id}.png',
        f'{carpeta}/roc_{task_id}.png',
    ]
    reporte_path = f'uploads/reporte_{task_id}.pdf'
    generar_reporte(df, graficas, reporte_path)
    return send_file(reporte_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)