import numpy as np  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from skimage.transform import resize  # type: ignore
from skimage.feature import hog  # type: ignore
from pathlib import Path
from PIL import Image  # type: ignore
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

def extract_features(images, target_size=(128, 128)):
    features = []
    for image in images:
        resized_image = resize(image, target_size, anti_aliasing=True)  # Ensure resizing consistency

        if len(resized_image.shape) == 3:
            resized_image = np.mean(resized_image, axis=2)  # Convert to grayscale if 3 channels

        hog_features = hog(resized_image,
                           pixels_per_cell=(16, 16),
                           cells_per_block=(2, 2),
                           feature_vector=True)
        features.append(hog_features)
    return np.array(features)


def prediction_with_single_image(image_path, model, scaler):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_array = np.array(img)

    features = extract_features([img_array])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    return prediction


def plot_result(image_path, prediction, true_label=None):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')  

    if true_label is not None:
        if prediction == true_label:
            plt.title(f'Prediction: {prediction} (Correct)', color='green')
        else:
            plt.title(f'Prediction: {prediction} (Incorrect)', color='red')
    else:
        plt.title(f'Prediction: {prediction}', color='blue')

    plt.show()


if __name__ == '__main__':
    svm_model = pickle.load(open('./Models/svm_classifier.pkl', 'rb'))
    scaler = pickle.load(open('./Models/scaler.pkl', 'rb'))

    image_path = 'aloevera.jpg'
    true_label = 'aloevera'  

    prediction = prediction_with_single_image(image_path, svm_model, scaler)
    print(f'Prediction: {prediction}')

    plot_result(image_path, prediction, true_label)
