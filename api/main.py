from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
# import tensorflow as tf

app = FastAPI()


@app.get("/")
async def ping():
    return 'hey'


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    return

if __name__ == "__main__":
    uvicorn.run(app, host='losthost', port=8000)
