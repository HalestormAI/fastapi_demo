from dataclasses import dataclass


@dataclass
class ClassificationResponse:
    class_name: str
    confidence: float