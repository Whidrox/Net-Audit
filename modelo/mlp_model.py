import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

def detectar_anomalias(X):
    max_muestras = 3000
    if len(X) > max_muestras:
        indices = np.random.choice(len(X), max_muestras, replace=False)
        X_muestra = X[indices]
    else:
        X_muestra = X
        indices = np.arange(len(X))

    # Clustering para generar etiquetas
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=5, max_iter=100)
    etiquetas_muestra = kmeans.fit_predict(X_muestra)
    etiquetas = kmeans.predict(X)

    # Split para evaluar modelos de forma diferente
    X_train, X_test, y_train, y_test = train_test_split(
        X_muestra, etiquetas_muestra, test_size=0.2, random_state=42
    )

    # MLP - más capas, más complejo
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=300,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X)
    prob_mlp = mlp.predict_proba(X)

    # Random Forest - diferente enfoque
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=24,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X)
    prob_rf = rf.predict_proba(X)

    # PCA para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_muestra)

    return {
        'etiquetas': etiquetas_muestra,
        'pred_mlp': pred_mlp,
        'prob_mlp': prob_mlp,
        'pred_rf': pred_rf,
        'prob_rf': prob_rf,
        'X_pca': X_pca,
        'X_test': X_test,
        'y_test': y_test,
        'kmeans': kmeans,
        'mlp': mlp,
        'rf': rf
    }

def mapear_etiqueta(pred):
    mapa = {0: 'Normal', 1: 'Sospechoso', 2: 'Posible Ataque'}
    return mapa.get(int(pred), 'Desconocido')

def obtener_metricas(etiquetas, pred_mlp, pred_rf, prob_mlp, prob_rf):
    n = min(len(etiquetas), len(pred_mlp), len(pred_rf))
    etiquetas = etiquetas[:n]
    pred_mlp = pred_mlp[:n]
    pred_rf = pred_rf[:n]
    prob_mlp = prob_mlp[:n]
    prob_rf = prob_rf[:n]

    clases = ['Normal', 'Sospechoso', 'Posible Ataque']

    reporte_mlp = classification_report(etiquetas, pred_mlp,
                                        target_names=clases,
                                        output_dict=True,
                                        zero_division=0)
    reporte_rf = classification_report(etiquetas, pred_rf,
                                       target_names=clases,
                                       output_dict=True,
                                       zero_division=0)

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