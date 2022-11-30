import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

from check import glcm_feature_extraction
from cnn.read_image_file import read_imagefile, predict

app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {"message": "Image must be jpg or png format!", "data": '', "assumption": ''}

    contents = await file.read()
    inp = np.asarray(bytearray(contents), dtype=np.uint8)

    img = cv2.imdecode(inp, cv2.IMREAD_COLOR)

    image = read_imagefile(contents)
    prediction = predict(image)

    result = glcm_feature_extraction(img)

    return {"message": "Berhasil", "data": result, "assumption": prediction}
