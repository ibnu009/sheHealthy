from fastapi import FastAPI

from check import cnn_classification

app = FastAPI()


@app.get("/my-first-api")
async def hello():
    ppp = cnn_classification()
    return {"message": ppp}
