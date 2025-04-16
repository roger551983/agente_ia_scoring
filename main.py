from typing import List
from fastapi import FastAPI
from matplotlib.font_manager import json_dump
import pandas as pd
from pydantic import BaseModel
from arbol_decision import ArbolDecisionScoring
import json 

class Cliente(BaseModel):
    descargaPDF: str
    clicWhatsApp: str
    visitaPagina: str
    esCliente: str

class ClientesRequest(BaseModel):
    clientes: List[Cliente]

app=FastAPI()

@app.post("/predecir_segmentacion")
def predecir_segmento(data: ClientesRequest):
    arbol = ArbolDecisionScoring()
    df_entrenamiento = arbol.getData()
    modelo= arbol.entrenar_modelo(df_entrenamiento)
     # Convertimos la lista de entrada a DataFrame
    df_input = pd.DataFrame([cliente.model_dump() for cliente in data.clientes])
    predicciones = arbol.predecir_segmentacion(modelo, df_input)
    return {"predicciones": predicciones}