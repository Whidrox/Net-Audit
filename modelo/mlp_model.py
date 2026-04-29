import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

def detectar_anomalias(X):
    # Clustering para generar etiquetas
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    etiquetas = kmeans.fit_predict(X)

    # Modelo MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X, etiquetas)
    pred_mlp = mlp.predict(X)
    prob_mlp = mlp.predict_proba(X)

    # Modelo Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, etiquetas)
    pred_rf = rf.predict(X)
    prob_rf = rf.predict_proba(X)

    # PCA para visualización de clusters
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    return {
        'etiquetas': etiquetas,
        'pred_mlp': pred_mlp,
        'prob_mlp': prob_mlp,
        'pred_rf': pred_rf,
        'prob_rf': prob_rf,
        'X_pca': X_pca,
        'kmeans': kmeans,
        'mlp': mlp,
        'rf': rf
    }

def mapear_etiqueta(pred):
    mapa = {0: 'Normal', 1: 'Sospechoso', 2: 'Posible Ataque'}
    return mapa.get(int(pred), 'Desconocido')

def obtener_metricas(etiquetas, pred_mlp, pred_rf, prob_mlp, prob_rf):
    clases = ['Normal', 'Sospechoso', 'Posible Ataque']

    # Reportes
    reporte_mlp = classification_report(etiquetas, pred_mlp,
                                        target_names=clases, output_dict=True)
    reporte_rf = classification_report(etiquetas, pred_rf,
                                       target_names=clases, output_dict=True)

    # Matrices de confusión
    cm_mlp = confusion_matrix(etiquetas, pred_mlp)
    cm_rf = confusion_matrix(etiquetas, pred_rf)

    # ROC (binarizado)
    y_bin = label_binarize(etiquetas, classes=[0, 1, 2])

    roc_mlp = {}
    roc_rf = {}
    for i, clase in enumerate(clases):
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob_mlp[:, i])
        roc_mlp[clase] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                          'auc': round(auc(fpr, tpr), 3)}
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob_rf[:, i])
        roc_rf[clase] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                         'auc': round(auc(fpr, tpr), 3)}

    return {
        'reporte_mlp': reporte_mlp,
        'reporte_rf': reporte_rf,
        'cm_mlp': cm_mlp,
        'cm_rf': cm_rf,
        'roc_mlp': roc_mlp,
        'roc_rf': roc_rf
    }