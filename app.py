from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
# import logging

app = FastAPI()

# logging.basicConfig(level=logging.INFO)

# Load your model
model = load_model("content\my_model")



@app.post("/predict/")
async def predict(file: UploadFile ):
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
        result = 'Not_Alzheimer'

    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)