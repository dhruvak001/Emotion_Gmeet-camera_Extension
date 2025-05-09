# Real-time Facial Emotion Recognition System

This project implements a real-time facial emotion recognition system using deep learning techniques. The system can detect and classify facial expressions in real-time using computer vision and deep learning models.

## Project Structure

```
DL_Project/
├── model_training/                 # Model training notebooks
│   ├── Transfer_Learning_approach.ipynb
│   ├── cnn_based_classification.ipynb
│   └── transformer_based.ipynb
├── camera_video_model_integration/ # Real-time implementation
├── emotion_meet_extension/        # Extension for video conferencing
└── DeepLearning_Report.pdf        # Project documentation
```

## Features

- Real-time facial emotion detection and classification
- Multiple deep learning approaches:
  - CNN-based classification
  - Transfer Learning
  - Transformer-based architecture
- Integration with video conferencing platforms
- Support for multiple emotion categories

## Technical Details

### Models Implemented

1. **CNN-based Classification**
   - Custom CNN architecture for emotion recognition
   - Implemented in `cnn_based_classification.ipynb`

2. **Transfer Learning**
   - Utilizes pre-trained models for improved performance
   - Implementation in `Transfer_Learning_approach.ipynb`

3. **Transformer-based Architecture**
   - Modern transformer-based approach for emotion recognition
   - Details in `transformer_based.ipynb`

### Real-time Implementation

The system uses OpenCV for real-time video capture and processing, integrated with the trained deep learning models for emotion recognition.

## Getting Started

### Prerequisites

- Python 3.8+
- OpenCV
- PyTorch or TensorFlow
- CUDA (for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd DL_Project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Train the models using the provided Jupyter notebooks in the `model_training` directory
2. Run the real-time implementation:
```bash
python camera_video_model_integration/main.py
```

## Research References

This project is based on several research papers and implementations:

1. Real-time Facial Emotion Recognition using Deep Learning and OpenCV
2. Facial Expression Recognition with Visual Transformers and Attentional Selective Fusion
3. Real Time Emotion Analysis Using Deep Learning for Education, Entertainment, and Beyond
4. Facial expression recognition via ResNet-50
5. A Comparative Analysis of CNNs and ResNet50 for Facial Emotion Recognition

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the researchers and developers whose work has been referenced in this project
- Special thanks to the open-source community for their valuable contributions
