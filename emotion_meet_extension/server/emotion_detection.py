import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

class EmotionDetector:
    def __init__(self, model_path='best_model_weights.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_height = 224
        self.img_width = 224
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_classes = ["Anger", "Disgust", "Fear", "Happiness", 
                              "Sadness", "Surprise", "Neutral", "Contempt"]

    def load_model(self, model_path):
        class EmotionCNN(nn.Module):
            def __init__(self, num_classes=8):
                super(EmotionCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(32),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(128),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(2, 2)
                )
                feature_map_size = (224 // 16) * (224 // 16)  # Hardcoded calculation
                classifier_input_dim = 256 * feature_map_size
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(classifier_input_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        model = EmotionCNN(num_classes=8).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def detect_emotions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (self.img_width, self.img_height))
            
            tensor = self.transform(face_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, preds = torch.max(probs, 1)
            
            results.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "emotion": self.emotion_classes[preds.item()],
                "confidence": confidence.item()
            })
        
        return results