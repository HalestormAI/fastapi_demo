from dataclasses import dataclass
from fastapi import FastAPI
from fastapi import UploadFile
import uvicorn

from .resnet import ResNetClassifier
from .models.item import ItemResponse, generate_random_items
from .models.classification import ClassificationResponse


app = FastAPI()

classifier = ResNetClassifier()
classifier.load()


@dataclass
class Message:
    message: str


@app.get("/", response_model=Message)
async def root():
    return {"message": "Hello World"}


@app.get("/items/{num_items}", response_model=ItemResponse)
async def item_list(num_items: int):
    return {"items": generate_random_items(num_items)}


@app.post("/classify", response_model=ClassificationResponse)
async def classify(input_image: UploadFile):
    preds, confidence = classifier.classify(input_image)
    return {"class_name": str(preds), "confidence": confidence}


if __name__ == "__main__":
    config = uvicorn.Config('simple:app', port=5000,
                            log_level="info", reload=True)
    server = uvicorn.Server(config)
    server.run()
