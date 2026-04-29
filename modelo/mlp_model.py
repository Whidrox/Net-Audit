import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

def detectar_anomalias(X):
    # Usar muestra si hay muchos datos
    max_muestras = 3000
    if len(X) > max_muestras:
        indices = np.random.choice(len(X), max_muestras, replace=False)
        X_muestra = X[indices]
    else:
        X_muestra = X
        indices = np.arange(len(X))

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=5, max_iter=100)
    etiquetas_muestra = kmeans.fit_predict(X_muestra)

    # Predecir etiquetas para todos
    etiquetas = kmeans.predict(X)

    # MLP optimizado
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_muestra, etiquetas_muestra)
    pred_mlp = mlp.predict(X)
    prob_mlp = mlp.predict_proba(X)

    # Random Forest optimizado
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_muestra, etiquetas_muestra)
    pred_rf = rf.predict(X)
    prob_rf = rf.predict_proba(X)

    # PCA solo sobre muestra
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_muestra)

    return {
        'etiquetas': etiquetas_muestra,
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
    # Alinear tamaños
    n = min(len(etiquetas), len(pred_mlp), len(pred_rf))
    etiquetas = etiquetas[:n]
    pred_mlp = pred_mlp[:n]
    pred_rf = pred_rf[:n]
    prob_mlp = prob_mlp[:n]
    prob_rf = prob_rf[:n]

    clases = ['Normal', 'Sospechoso', 'Posible Ataque']

    reporte_mlp = classification_report(etiquetas, pred_mlp,
                                        target_names=clases, output_dict=True)
    reporte_rf = classification_report(etiquetas, pred_rf,
                                       target_names=clases, output_dict=True)

    cm_mlp = confusion_matrix(etiquetas, pred_mlp)
    cm_rf = confusion_matrix(etiquetas, pred_rf)

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