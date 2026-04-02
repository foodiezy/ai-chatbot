import os
import tensorflow as tf
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog

def load_vision_model():
    model_path = "animal_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def predict_image(model, image_path):
    try:

        img = Image.open(image_path)
        img = img.resize((150, 150))
        

        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = np.array(img).astype("float32")

        img_array = np.expand_dims(img_array, axis=0)


        prediction = model.predict(img_array, verbose=0)
        

        if prediction[0][0] > 0.5:
            return "This image looks like a dog."
        else:
            return "This image looks like a cat."
            
    except Exception as e:
        return f"Could not process image: {e}"

def classify_user_image(model):
    if not model:
        return "The CNN model not found. Please run train_cnn.py first."


    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.lift()
    
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    
    if not file_path:
        return "No file selected."
        
    return predict_image(model, file_path)
