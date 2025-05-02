import csv

data_entrenamiento={
    "descargaPDF": [],
    "clicWhatsApp": [],
    "visitaPagina": [],
    "esCliente": [],
    "segmentacion": []
}

def cargar_datos()->dict :
     with open('Data/entrenamiento_prospeccion_leads.csv', newline='', encoding='utf-8') as archivo:
          reader = csv.DictReader(archivo)
          for fila in reader:
           for clave in data_entrenamiento:
               data_entrenamiento[clave].append(fila[clave])
     return data_entrenamiento