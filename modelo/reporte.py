from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os

def generar_reporte(df_resultados, graficas_paths, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elementos = []

    # Título
    elementos.append(Paragraph("Reporte de Auditoría de Tráfico de Red", styles['Title']))
    elementos.append(Spacer(1, 12))

    # Resumen
    total = len(df_resultados)
    normales = len(df_resultados[df_resultados['Clasificacion_MLP'] == 'Normal'])
    sospechosos = len(df_resultados[df_resultados['Clasificacion_MLP'] == 'Sospechoso'])
    ataques = len(df_resultados[df_resultados['Clasificacion_MLP'] == 'Posible Ataque'])

    elementos.append(Paragraph("Resumen del Análisis", styles['Heading2']))
    elementos.append(Paragraph(f"Total de paquetes analizados: {total}", styles['Normal']))
    elementos.append(Paragraph(f"Paquetes normales: {normales}", styles['Normal']))
    elementos.append(Paragraph(f"Paquetes sospechosos: {sospechosos}", styles['Normal']))
    elementos.append(Paragraph(f"Posibles ataques detectados: {ataques}", styles['Normal']))
    elementos.append(Spacer(1, 12))

    # Gráficas
    elementos.append(Paragraph("Gráficas de Análisis", styles['Heading2']))
    for path in graficas_paths:
        if os.path.exists(path):
            elementos.append(Image(path, width=400, height=250))
            elementos.append(Spacer(1, 12))

    # Tabla de resultados (primeros 50)
    elementos.append(Paragraph("Detalle de Paquetes (primeros 50)", styles['Heading2']))
    elementos.append(Spacer(1, 6))

    datos = [df_resultados.columns.tolist()] + df_resultados.head(50).values.tolist()
    tabla = Table(datos, repeatRows=1)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    elementos.append(tabla)

    doc.build(elementos)
    return output_path