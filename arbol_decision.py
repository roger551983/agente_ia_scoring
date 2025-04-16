from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import data_entrenamiento as d

#clientes entrenamiento

class ArbolDecisionScoring:

    def getData(self):
        clientes_dict = d.cargar_datos()
        return pd.DataFrame(clientes_dict)
    
    def mapear_respuesta(self,respuesta,tipo):
        puntajes = {
            "descargaPDF": {"Si": 10, "No": 0},
            "clicWhatsApp": {"Si": 5, "No": 0},
            "visitaPagina": {"Si": 5, "No": 0},
            "esCliente": {"Si": 5, "No": 0}
        }
        return puntajes[tipo].get(respuesta, 0)
    
    def calcular_puntajes(self,df):
       
       df["puntajeDescarga"] = df["descargaPDF"].apply(lambda x:self.mapear_respuesta( x,tipo="descargaPDF"))
       df["puntajeClick"] = df["clicWhatsApp"].apply(lambda x: self.mapear_respuesta(x, tipo="clicWhatsApp"))
       df["puntajeVisita"] = df["visitaPagina"].apply(lambda x: self.mapear_respuesta(x, tipo="visitaPagina"))
       df["puntajeCliente"] = df["esCliente"].apply(lambda x: self.mapear_respuesta(x, tipo="esCliente"))
       df["puntajeTotal"] = df[["puntajeDescarga", "puntajeClick", "puntajeVisita", "puntajeCliente"]].sum(axis=1)
       return df
    def determinar_segmentacion(self,row):
        if row["esCliente"] == "Si" and row["puntajeTotal"] >= 20:
            return "decision"
        elif row["esCliente"] == "Si" and row["puntajeTotal"] < 20:
            return "consideracion"
        elif row["esCliente"] == "No" and row["puntajeTotal"] <= 10:
            return "descubrimiento"
        elif row["esCliente"] == "No" and row["puntajeTotal"] > 10:
            return "consideracion"
        else:
            return "Inexistente"
    def entrenar_modelo(self,df):
        df_puntaje= self.calcular_puntajes(df)
        df_puntaje["segmentacion"] = df_puntaje.apply(self.determinar_segmentacion, axis=1)
        segmentacion_map = {"decision": 0, "consideracion": 1, "descubrimiento": 2, "Inexistente": 3}
        df_puntaje["segmentacionNum"] = df["segmentacion"].map(segmentacion_map)
        X = df_puntaje[["puntajeDescarga", "puntajeClick", "puntajeVisita", "puntajeCliente", "puntajeTotal"]]
        y=  df_puntaje["segmentacionNum"]
        modelo = DecisionTreeClassifier()
        modelo.fit(X,y)
        return modelo
    
    def predecir_segmentacion(self,modelo_entrenado,data_predict):
        df_puntaje= self.calcular_puntajes(data_predict)
        segmentacion_map = {0: "decision", 1: "consideracion", 2: "descubrimiento", 3: "Inexistente"}
        X_predict= df_puntaje[["puntajeDescarga", "puntajeClick", "puntajeVisita", "puntajeCliente", "puntajeTotal"]]
         # Predice y mapea la predicción al segmento
        prediction = modelo_entrenado.predict(X_predict)
        return  [segmentacion_map[i]for i in list(prediction)]
        
        




 



