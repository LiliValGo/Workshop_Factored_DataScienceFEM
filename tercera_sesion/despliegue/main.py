"""vamos a importar las  dependencias necesariar para usar FastAPI, el preprocesador y el modelo. 
También vamos a crear nuestra API, la cual tendrá el nombre de "Taxi Trips Duration Predictor"."""


import numpy as np
import pandas as pd
import dill
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse, StreamingResponse 
"""StreamingResponse, recibe como entrada un archivo y devolverá por tanto un archivo. No guarda los archivos en disco sino que en el Browser recibe el archivo""" 

from io import BytesIO
"""BytesIO de la librería io, la cual toma un objeto bytes que está en memoria en python e imita el comportamiento de un archivo, de lo que Python considera un objeto tipo archivo, el cual tiene métodos de "read" y "write"."""

from pydantic import BaseModel #clase de la cual heredamos para crear especificaciones del JSON

from transformers import TransformerFechas, TransformerDistancia, TransformerVelocidad

app = FastAPI(title="Taxi Trips Duration Predictor")

# import the preprocessor
with open("preprocesser.pkl", "rb") as f:
    preprocessor = dill.load(f)
# import the model
with open("lr_model.pkl", "rb") as f:
    model = dill.load(f)

    
# Primer Endpoint
"""Get Endpoint:Las solicitudes GET son los métodos más comunes y más utilizados en API y sitios web. El método GET se utiliza para recuperar datos de un servidor en el recurso especificado.

El método GET puede recibir parámetros a través de parámetros "path" o parámetros "query"""

@app.get("/", response_class=JSONResponse)
def get_funct(
    vendor_id: int,
    pickup_datetime: str,
    passenger_count: int,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    pickup_borough: str,
    dropoff_borough: str,
):
    """Serves predictions given query parameters specifying the taxi trip's
    features from a single example.

    Args:
        vendor_id (int): a code indicating the provider associated with the trip record
        pickup_datetime (str): date and time when the meter was engaged
        passenger_count (float): the number of passengers in the vehicle
        (driver entered value)
        pickup_longitude (float): the longitude where the meter was engaged
        pickup_latitude (float): the latitude where the meter was engaged
        dropoff_longitude (float): the longitude where the meter was disengaged
        dropoff_latitude (float): the latitude where the meter was disengaged
        pickup_borough (str): the borough where the meter was engaged
        dropoff_borough (str): the borough where the meter was disengaged

    Returns:
        [JSON]: model prediction for the single example given
    """
    df = pd.DataFrame(
        [
            [
                vendor_id,
                pickup_datetime,
                passenger_count,
                pickup_longitude,
                pickup_latitude,
                dropoff_longitude,
                dropoff_latitude,
                pickup_borough,
                dropoff_borough,
            ]
        ],
        columns=[
            "vendor_id",
            "pickup_datetime",
            "passenger_count",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "pickup_borough",
            "dropoff_borough",
        ],
    )
    prediction = model.predict(preprocessor.transform(df))
    return {
        "features": {
            "vendor_id": vendor_id,
            "pickup_datetime": pickup_datetime,
            "passenger_count": passenger_count,
            "pickup_longitude": pickup_longitude,
            "pickup_latitude": pickup_latitude,
            "dropoff_longitude": dropoff_longitude,
            "dropoff_latitude": dropoff_latitude,
            "pickup_borough": pickup_borough,
            "dropoff_borough": dropoff_borough,
        },
        "prediction": list(prediction)[0],
    }


if __name__ == "__main__":
    import uvicorn

    # For local development:
    uvicorn.run("main:app", port=3000, reload=True)
    
"""reload=True permite que cada vez que se guarde el archivo main.py, la API se actualice automáticamente, sin necesidad de cerrar la API y volverla a reiniciar cada vez que se haga un cambio"""



# Segundo Endpoint
"""Post Endpoint: En los servicios web, las solicitudes POST se utilizan para enviar datos al servidor API para crear o actualizar un recurso. Los datos enviados al servidor se almacenan en el cuerpo de la solicitud HTTP"""

#clase TaxiTrip, que hereda de BaseModel
class TaxiTrip(BaseModel): 
    vendor_id: int
    pickup_datetime: str
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    pickup_borough: str
    dropoff_borough: str
        
"""TaxiTrip va a ser un objeto con 9 parámetros que 
describen un viaje en taxi"""


#definimos que el parámetro que entra es un taxitrip de tipo TaxiTrip.

@app.post("/json", response_class=JSONResponse)
def post_json(taxitrip: TaxiTrip):
    """Serves predictions given a request body specifying the taxis trip's features
    from a single example.

    Args:
        taxitrip (TaxiTrip): request body of type `TaxiTrip` with the
        attributes: vendor_id, pickup_datetime, passenger_count, pickup_longitude,
        pickup_latitude, dropoff_longitude, dropoff_latitude, pickup_borough and
        dropoff_borough

    Returns:
        [JSON]: model prediction for the single example given
    """
    vendor_id = taxitrip.vendor_id
    pickup_datetime = taxitrip.pickup_datetime
    passenger_count = taxitrip.passenger_count
    pickup_longitude = taxitrip.pickup_longitude
    pickup_latitude = taxitrip.pickup_latitude
    dropoff_longitude = taxitrip.dropoff_longitude
    dropoff_latitude = taxitrip.dropoff_latitude
    pickup_borough = taxitrip.pickup_borough
    dropoff_borough = taxitrip.dropoff_borough

    df = pd.DataFrame(
        [
            [
                vendor_id,
                pickup_datetime,
                passenger_count,
                pickup_longitude,
                pickup_latitude,
                dropoff_longitude,
                dropoff_latitude,
                pickup_borough,
                dropoff_borough,
            ]
        ],
        columns=[
            "vendor_id",
            "pickup_datetime",
            "passenger_count",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "pickup_borough",
            "dropoff_borough",
        ],
    )
    prediction = model.predict(preprocessor.transform(df))
    return {
        "features": {
            "vendor_id": vendor_id,
            "pickup_datetime": pickup_datetime,
            "passenger_count": passenger_count,
            "pickup_longitude": pickup_longitude,
            "pickup_latitude": pickup_latitude,
            "dropoff_longitude": dropoff_longitude,
            "dropoff_latitude": dropoff_latitude,
            "pickup_borough": pickup_borough,
            "dropoff_borough": dropoff_borough,
        },
        "prediction": list(prediction)[0],
    }

"""POST Endpoint"""

#Tercer Endpoint
@app.post("/file", response_class=StreamingResponse)
def post_file(file: bytes = File(...)):
    """Serves predictions given a CSV file with no header and seven columns
    specifying each taxi trip's features in the order vendor_id, pickup_datetime,
    passenger_count, pickup_longitude,pickup_latitude, dropoff_longitude and
    dropoff_latitude, pickup_borough and dropoff_borough

    Args:
        file (bytes, optional): bytes from a CSV file as described above.
         Defaults to File(...), but to receive a file is required.

    Returns:
        [StreamingResponse]: Returns a streaming response with a new CSV file that contains
        a column with the predictions.
    """
    # Decode the bytes as text and split the lines:
    input_lines = file.decode().splitlines()

    # Split each line as a list of the three features:
    X = [p.split(",") for p in input_lines]
    predictions = []
    for x in X:
        vendor_id = int(x[0])
        pickup_datetime = str(x[1])
        passenger_count = float(x[2])
        pickup_longitude = float(x[3])
        pickup_latitude = float(x[4])
        dropoff_longitude = float(x[5])
        dropoff_latitude = float(x[6])
        pickup_borough = str(x[7])
        dropoff_borough = str(x[8])
        df = pd.DataFrame(
            [
                [
                    vendor_id,
                    pickup_datetime,
                    passenger_count,
                    pickup_longitude,
                    pickup_latitude,
                    dropoff_longitude,
                    dropoff_latitude,
                    pickup_borough,
                    dropoff_borough,
                ]
            ],
            columns=[
                "vendor_id",
                "pickup_datetime",
                "passenger_count",
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "pickup_borough",
                "dropoff_borough",
            ],
        )
        # Get predictions for each taxi trip:
        prediction = model.predict(preprocessor.transform(df))
        predictions.append(prediction)

    # Append the prediction to each input line:
    output = [line + "," + str(pred[0]) for line, pred in zip(input_lines, predictions)]
    # Join the output as a single string:
    output = "\n".join(output)
    # Encode output as bytes:
    output = output.encode()

    # The kind is text, the extension is csv
    return StreamingResponse(
        BytesIO(output),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment;filename="prediction.csv"'},
    )

"""Cambios para hacer en nuestro archivo del backend, main.py
1. Incluir CORSMiddleware: CORS un mecanismo para controlar quién puede acceder a nuestra API, quién puede realizar solicitudes y obtener respuestas."""
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""Cambiar el puerto Localhost"""
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=3000, reload=True)
"""Actualmente la aplicación intentará ejecutarse en enlocalhost en el puerto 3000. Eso no va a funcionar. Porque el host local del contenedor de la ventana acoplable no estará expuesto al host que ejecuta el contenedor. Incluso si lo fuera, no sería apropiado porque el puerto que exponen nuestras imágenes de la ventana acoplable es el puerto 80"""

if __name__ == "__main__":
    import uvicorn

    # For local development:
    # uvicorn.run("main:app", port=3000, reload=True)

    # for docker deployment:
    uvicorn.run("main:app", host="0.0.0.0", port=80)