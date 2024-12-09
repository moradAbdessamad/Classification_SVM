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


def load_dataset(data_path, subset='train', target_size=(128, 128)):
    data_dir = Path(data_path) / subset
    images, labels = [], []
    classes = ['aloevera', 'coconut', 'pineapple', 'spinach']

    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            continue
        for image_path in class_dir.glob('*.[jp][np][ge]*'):  # Adjusted glob pattern for image files
            try:
                img = Image.open(image_path)
                img = img.convert('RGB')
                
                # Resize image to target size (e.g., 128x128)
                img = img.resize(target_size)
                img_array = np.array(img)
                
                images.append(img_array)
                labels.append(class_name)
                
            except Exception as e:
                print(f'Error processing {image_path}: {e}')
    return np.array(images), np.array(labels)


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


def train_and_evaluate_svm():
    print("Loading training dataset...")
    X_train, y_train = load_dataset('./data', subset='train')

    print("Loading testing dataset...")
    X_test, y_test = load_dataset('./data', subset='test')

    print("Extracting features from training dataset...")
    train_features = extract_features(X_train)

    print("Extracting features from testing dataset...")
    test_features = extract_features(X_test)

    print("Scaling features...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    print("Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(train_features_scaled, y_train)

    print("Evaluating SVM...")
    y_pred = svm.predict(test_features_scaled)
    print(classification_report(y_test, y_pred))

    # Save the model and scaler
    if not os.path.exists('Models'):
        os.makedirs('Models')

    model_path = os.path.join('Models', 'svm_classifier.pkl')
    scaler_path = os.path.join('Models', 'scaler.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(svm, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model and scaler saved at {model_path} and {scaler_path}")
    return svm, scaler


def predication_with_single_image(image_path, model, scaler):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_array = np.array(img)

    features = extract_features([img_array])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    return prediction


def load_model_and_scaler(model_path='Models/svm_classifier.pkl', scaler_path='Models/scaler.pkl'):
    """ Load the trained model and scaler from the specified paths. """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


if __name__ == '__main__':
    svm_model, scaler = train_and_evaluate_svm()

    # For making predictions
    image_path = 'testImage.jpg'
    prediction = predication_with_single_image(image_path, svm_model, scaler)
    print(f'Prediction: {prediction}')
