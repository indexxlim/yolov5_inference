import json
import base64
from PIL import Image
import io
import torch
from loguru import logger





@torch.no_grad()
def inference(model, data):
    logger.info("Run yolo-v5 model")

    threshold = data.threshold
    binary_image = data.binary_image
    
    buf = io.BytesIO(base64.b64decode(binary_image))
    threshold = float(threshold)
    model.conf = threshold
    image = Image.open(buf)
    yolo_results_json = model(image).pandas().xyxy[0].to_dict(orient='records')

    encoded_result = []
    for result in yolo_results_json:
        encoded_result.append({
            'confidence': result['confidence'],
            'label': result['name'],
            'points': [
                result['xmin'],
                result['ymin'],
                result['xmax'],
                result['ymax']
            ],
            'type': 'rectangle'
        })
    print({'result': encoded_result})
    return {'result': encoded_result}