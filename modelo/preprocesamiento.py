import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def cargar_y_preprocesar(filepath):
    # Cargar CSV
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Eliminar columnas no útiles para el modelo
    df = df.drop(columns=['No.', 'Info'], errors='ignore')

    # Convertir Time a numérico
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

    # Extraer si la IP es IPv6
    df['Source_IPv6'] = df['Source'].apply(lambda x: 1 if ':' in str(x) else 0)
    df['Dest_IPv6'] = df['Destination'].apply(lambda x: 1 if ':' in str(x) else 0)

    # Detectar IPs de broadcast/multicast
    df['Es_Multicast'] = df['Destination'].apply(
        lambda x: 1 if str(x).startswith('224.') or str(x).startswith('239.')
        or str(x).startswith('ff') else 0
    )

    # Eliminar columnas de IP originales
    df = df.drop(columns=['Source', 'Destination'], errors='ignore')

    # Codificar Protocol
    le = LabelEncoder()
    df['Protocol'] = le.fit_transform(df['Protocol'].astype(str))

    # Convertir Length a numérico
    df['Length'] = pd.to_numeric(df['Length'], errors='coerce')

    # Eliminar filas con valores nulos
    df = df.dropna()

    # Escalar features
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    return X, df.columns.tolist(), le, scaler