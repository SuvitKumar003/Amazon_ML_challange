import os
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from student_resource_3.src.utils import download_images  # Assuming you have this function for downloading images
from student_resource_3.src.constants import ALLOWED_UNITS, entity_unit_map  # Import constants from the same directory

# Define the predictor function
def predictor(model, img_url, entity_name):
    try:
        # Download the image from the URL
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))  # Resize to the required input size of EfficientNet

        # Convert image to numpy array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict using the model
        prediction = model.predict(img_array)
        predicted_value = prediction[0][0]  # Assuming a single output, modify as needed

        # Determine allowed units for the entity
        allowed_units = entity_unit_map.get(entity_name, {"unit"})  # Default to "unit" if not found

        # Choose the appropriate unit based on the entity name (replace with your logic)
        unit = choose_unit(entity_name, allowed_units)

        if unit not in allowed_units:
            return ""  # Return an empty string if unit is not allowed

        # Format prediction
        return f"{predicted_value:.2f} {unit}"
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        return ""  # Return an empty string if there is an error

# Function to choose the appropriate unit (replace with your logic)
def choose_unit(entity_name, allowed_units):
    # Example implementation:
    if entity_name == "length":
        return "cm"  # Assuming "cm" is a valid unit for length
    elif entity_name == "weight":
        return "kg"  # Assuming "kg" is a valid unit for weight
    else:
        return "unit"  # Default to "unit" if no specific unit is found

# Main part of the code
if __name__ == "__main__":
    # Load the dataset
    DATASET_FOLDER = 'student_resource 3/dataset'
    test_file_path = os.path.join(DATASET_FOLDER, 'test.csv')

    # Check if the dataset file exists
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"{test_file_path} not found")

    # Read the test CSV file
    test = pd.read_csv(test_file_path)

    # Check if required columns exist in the CSV
    required_columns = ['image_link', 'group_id', 'entity_name']
    for column in required_columns:
        if column not in test.columns:
            raise KeyError(f"Column '{column}' not found in the dataset")

    # Load EfficientNetB0 model (pre-trained on ImageNet)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add classification layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='linear')  # Adjust the output layer as needed
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    # Load trained weights if available
    model_weights_path = 'path_to_trained_weights.h5'
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)
    else:
        raise FileNotFoundError(f"Model weights file {model_weights_path} not found")

    # Loop through the test set and apply predictions
    test['prediction'] = test.apply(lambda row: predictor(model, row['image_link'], row['entity_name']), axis=1)

    # Save predictions to a new CSV file
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)

    print(f"Predictions saved to {output_filename}")



    