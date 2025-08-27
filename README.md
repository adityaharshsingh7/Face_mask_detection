About
This project implements a Face Mask Detection system using deep learning, developed in Python with TensorFlow and OpenCV. It uses the MobileNetV2 architecture pretrained on ImageNet and fine-tuned on a labeled face mask dataset from Kaggle. The system detects faces in images and classifies whether the person is wearing a mask or not, providing prediction confidence percentages.

The solution is optimized for memory efficiency with batch loading via Keras' ImageDataGenerator to handle large image datasets in Google Colab without exceeding RAM limits. It supports model training, evaluation, and real-time mask detection on uploaded images.

This work serves as a practical example of transfer learning, image preprocessing, and real-time computer vision in healthcare safety applications.
