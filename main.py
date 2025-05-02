import json
from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel
from arbol_decision import ArbolDecisionScoring


class Cliente(BaseModel):
    nombre: str
    correo: str
    descargaPDF: str
    clicWhatsApp: str
    visitaPagina: str
    esCliente: str

class ClientesRequest(BaseModel):
    clientes: List[Cliente]

app=FastAPI()
app.mount("/static", StaticFiles(directory="Grafico_Dispersion"), name="static")

@app.post("/predecir_segmentacion")
def predecir_segmento(data: ClientesRequest):
    arbol = ArbolDecisionScoring()
    df_entrenamiento = arbol.getData()
    modelo= arbol.entrenar_modelo(df_entrenamiento)
     # Convertimos la lista de entrada a DataFrame
    df_input = pd.DataFrame([cliente.model_dump() for cliente in data.clientes])
    predicciones_lista = arbol.predecir_segmentacion(modelo, df_input)
    arbol.df_plot = df_input.copy()
    nombre_archivo_svg = arbol.CrearGraficoDispersion(predicciones_lista)
    url_grafico = f"http://localhost:8000/static/{nombre_archivo_svg}"
    print(list(predicciones_lista))
    resultados=[]
    for cliente, pred in zip(data.clientes, predicciones_lista):
        resultados.append({
            "nombre": cliente.nombre,
            "correo": cliente.correo,
            "segmento": pred
        })
   
    return {
            "predicciones": resultados,
            "grafico_url": url_grafico
            
            }

   
   