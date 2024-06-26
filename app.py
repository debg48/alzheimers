from fastapi import FastAPI, File, UploadFile, Request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import io
# import logging

app = FastAPI()

# logging.basicConfig(level=logging.INFO)

# Load your model
model = load_model("content\my_model")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Serve static files (like CSS) from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    # logging.info(f"Received file: {file.filename}")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((240, 240))  # Resize to 240x240

    # Preprocess the image
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = np.argmax(model.predict(img_array))
    # logging.info(f"Prediction: {prediction}")
    # if prediction
    print(prediction)

    if int(prediction) == 0:
        result = 'Alzheimer'
    else : 
        result = 'Not Alzheimer'

    return templates.TemplateResponse("result.html", {"request": request, "prediction": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)