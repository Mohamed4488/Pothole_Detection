import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from predict import predict_image_n, predict_image_s


app = FastAPI()



@app.post("/yolo11n")
async def predict_image_route(file: UploadFile = File(...)):
    
    image = Image.open(file.file)
    
    predicted_image = predict_image_n(image)
    
    buffer = io.BytesIO()
    predicted_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/yolo11s")
async def predict_image_route(file: UploadFile = File(...)):
    
    image = Image.open(file.file)
    
    predicted_image = predict_image_s(image)
    
    buffer = io.BytesIO()
    predicted_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/png")