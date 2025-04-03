import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

# Define input size and emotion classes
img_height, img_width = 224, 224
emotion_classes = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral", "Contempt"]

# Define CNN model
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
        feature_map_size = (img_height // 16) * (img_width // 16)
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

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EmotionCNN(num_classes=8).to(device)
weights_path = "/home/best_model_weights.pth"

if os.path.exists(weights_path):
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Weight file not found at: {weights_path}")
    exit(1)

model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_height, img_width), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default camera
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (img_width, img_height), interpolation=cv2.INTER_AREA)

        # Transform and predict
        face_tensor = transform(face_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(face_tensor)
            emotion_label = emotion_classes[torch.argmax(output).item()]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()