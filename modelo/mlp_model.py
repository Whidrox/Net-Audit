import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

def detectar_anomalias(X):
    """
    Como no tenemos etiquetas (normal/ataque), usamos KMeans
    para agrupar y luego MLP para clasificar los grupos
    """
    # Agrupar tráfico en clusters: normal, sospechoso, ataque
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    etiquetas = kmeans.fit_predict(X)

    # Entrenar MLP con esas etiquetas
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=42
    )
    mlp.fit(X, etiquetas)
    predicciones = mlp.predict(X)
    probabilidades = mlp.predict_proba(X)

    return predicciones, probabilidades, kmeans, mlp

def mapear_etiqueta(pred):
    mapa = {0: 'Normal', 1: 'Sospechoso', 2: 'Posible Ataque'}
    return mapa.get(pred, 'Desconocido')