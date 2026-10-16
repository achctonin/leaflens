
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from data import get_class_names


IMAGE_SIZE = (224, 224)


def predict_image(model_path, image_path):
    model_path = Path(model_path)
    image_path = Path(image_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the trained model.
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # Load the image as RGB and resize to 224x224.
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # Convert to float32, keeping pixel values from 0 to 255.
    image_array = np.asarray(image, dtype=np.float32)

    # Add the batch dimension: (1, 224, 224, 3).
    image_batch = np.expand_dims(image_array, axis=0)

    # Predict the disease class.
    predictions = model.predict(image_batch, verbose=0)[0]

    class_names = get_class_names()

    if len(predictions) != len(class_names):
        raise ValueError(
            "Model output count does not match the class list."
        )

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    print("\nLEAFLENS PREDICTION")
    print("----------------------------")
    print(f"Image: {image_path.name}")
    print(f"Model: {model_path.name}")
    print(f"Predicted disease: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nAll class probabilities:")
    for index in np.argsort(predictions)[::-1]:
        print(f"{class_names[index]}: {predictions[index] * 100:.2f}%")

    return predicted_class, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict mango leaf disease using LeafLens."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to a trained .keras model"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a mango leaf image"
    )

    args = parser.parse_args()

    predict_image(args.model, args.image)
