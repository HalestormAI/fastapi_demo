import json
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Callable, Dict, Optional

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import ToTensor

from PIL import Image

ROOT_PATH = Path(__file__).parent

class ResNetClassifier:
    def __init__(self):
        self.model: Optional[Callable] = None
        self.preprocessor: Optional[Callable] = None
        self.classes: Optional[Dict[int, str]] = None
    
    def load(self):
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.model.eval()

        self.classes = self._load_classes()

        self.preprocessor = weights.transforms()

    def _load_classes(self):
        classes = json.load(open(ROOT_PATH / "imagenet_classes.json"))
        return {int(k): v[1] for k, v in classes.items()}

    def _preprocess(self, image_file: SpooledTemporaryFile) -> torch.Tensor:
        pil_img = Image.open(image_file.file)
        img = ToTensor()(pil_img).unsqueeze(0)
        return self.preprocessor(img)

    def classify(self, image_file: SpooledTemporaryFile):
        if self.model is None:
            raise RuntimeError(
                "Model has not been loaded yet. Make sure to call 'load' first.")

        img = self._preprocess(image_file)
        outputs = self.model(img)
        logits = torch.softmax(outputs, 1).squeeze()
        pred_idx = torch.argmax(outputs, 1)
        return self.classes[pred_idx.item()], logits[pred_idx].item()
