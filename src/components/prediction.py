import torch
from model import PlantDiseaseResNet18
from utils import preprocess_image_bytes, PLANT_DISEASE_CLASSES
from src.config import MODEL_PATH


class PredictionPipeline:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()

    def load_model(self):
        model = PlantDiseaseResNet18(
            num_classes=len(PLANT_DISEASE_CLASSES)
        )

        ckpt = torch.load(MODEL_PATH, map_location=self.device)

        model.load_state_dict(
            ckpt['model_state'] if 'model_state' in ckpt else ckpt
        )

        model.to(self.device)
        model.eval()

        return model

    def predict(self, image_bytes):
        tensor = preprocess_image_bytes(image_bytes)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = int(torch.argmax(outputs, dim=1).item())

        return {
            "predicted_class": PLANT_DISEASE_CLASSES[pred_idx],
            "predicted_index": pred_idx,
            "confidence": float(probs[0][pred_idx])
        }