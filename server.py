from cv2 import threshold
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
from pydantic import BaseModel
from loguru import logger
import torch
from cus_base64 import Base64Bytes
from inference import inference


app = FastAPI()


class box(BaseModel):
    confidence: float
    label: str
    points: List[float]
    type: str
    

class responseJSON(BaseModel):
    result: List[box]   

        
class requestJSON(BaseModel):
    binary_image: Base64Bytes
    threshold: float = 0.5

    
    
logger.info("Init context... 0%")
if torch.cuda.is_available():
    device = 'cuda'
else:
    device= 'cpu'

#Load the DL model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
logger.info("Init context...100%")


@app.exception_handler(Exception)
async def unicorn_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=400,
        content={"error": type(exc).__name__,
                 "message": exc.args[0]},
    )


@app.post("/generate", response_model=responseJSON)
def generate(input_data: requestJSON):
    try:
        results = inference(
            model,
            input_data
        )
        return results
    except Exception as exc:
        raise RuntimeError(exc)
    

'''
    3. Encapsulated Route for configuration parameters (Do Not Modify)
'''
@app.get("/")
def api_info():
    return {
        "server_configs": 'yolov5',
        "request_json": requestJSON.schema(),
        "response_json": responseJSON.schema()
    }
