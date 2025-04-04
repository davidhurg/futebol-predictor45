from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("modelo/modelo_treinado.pkl")

@app.get("/prever")
def prever(gols_mandante: int, gols_visitante: int, ranking_mandante: float, ranking_visitante: float):
    entrada = [[gols_mandante, gols_visitante, ranking_mandante, ranking_visitante]]
    resultado = model.predict(entrada)[0]
    return {"previsão": int(resultado)}