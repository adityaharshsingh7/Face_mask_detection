# **Face Mask Detection Using Deep Learning**

## **Objective**

The aim of this project is to build an efficient face mask detection system that can classify whether a person is wearing a mask or not from an image. This has real-world applications in public health monitoring, especially during pandemics.

---

## **Tools and Technologies Used**

* **Languages**: Python
* **Libraries**:

  * TensorFlow / Keras
  * OpenCV
  * scikit-learn
  * Matplotlib
* **Dataset Source**: [Kaggle - andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

---

## **Dataset Description**

The dataset contains:

* Images of people **with mask**, **without mask**, and **mask worn incorrectly**
* Corresponding **annotation files (XML)** with class labels

Annotations were parsed to extract the label (class), and images were read and resized to a uniform shape (`128x128 pixels`) for processing.

---

## **Data Preprocessing**

1. **XML Parsing**: Custom function used to extract labels from XML files.
2. **Image Processing**:

   * Loaded and resized each image to 128x128 pixels.
   * Flattened images into 1D arrays suitable for feeding into a neural network.
3. **Label Encoding**: Converted categorical class labels to numerical format using LabelEncoder and one-hot encoding.
4. **Train-Test Split**: Dataset split into training and testing sets using `train_test_split()`.

---

## **Model Architecture**

A deep **Artificial Neural Network (ANN)** was constructed using TensorFlow/Keras. Key layers include:

* Dense layers with ReLU activations
* Dropout layers to prevent overfitting
* Batch Normalization for improved training performance
* Output layer with softmax activation for multi-class classification

The model was compiled with:

* **Loss Function**: Categorical Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy

EarlyStopping and ReduceLROnPlateau callbacks were used to enhance performance and prevent overfitting.

---

## **Evaluation Metrics and Results**

The model was evaluated on the test set using accuracy and visualized using plots (not yet extracted from full notebook). The performance will be quantitatively detailed after reviewing the final cells.

---

## **Live Detection System**

The project includes a live face mask detection system using OpenCV and Haar Cascades. This system:

* Captures frames from the webcam in real time
* Detects faces using Haar cascades
* Classifies each face as `mask`, `no mask`, or `incorrect mask` using the trained model
* Displays the result with bounding boxes and class labels on the video feed

---

## **Conclusion**

This project successfully demonstrates a complete workflow from dataset preparation and model training to real-time deployment. It offers an effective solution for face mask compliance monitoring in public places.

---

## **Future Work**

* Improve accuracy using convolutional neural networks (CNNs)
* Use bounding box annotations to train a face detection + classification model in one step
* Deploy the model using Flask or Streamlit for web-based applications


