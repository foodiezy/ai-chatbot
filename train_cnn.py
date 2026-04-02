import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import image_dataset_from_directory
import kagglehub
import os
import shutil

def train_and_save_model():
    print("Downloading Official Kaggle Cats and Dogs Dataset via KaggleHub...")
    
    try:
        path = kagglehub.dataset_download("tongpython/cat-and-dog")
    except Exception as e:
        print(f"Failed to download the dataset: {e}")
        return


    original_train = os.path.join(path, 'training_set', 'training_set')
    original_val = os.path.join(path, 'test_set', 'test_set')
    
    local_datasets = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'kaggle_images')
    local_train = os.path.join(local_datasets, 'train')
    local_val = os.path.join(local_datasets, 'validation')
    
    if not os.path.exists(local_train):
        print("Copying Kaggle images into your project 'datasets' folder (this might take a moment)...")
        os.makedirs(local_datasets, exist_ok=True)
        shutil.copytree(original_train, local_train)
        shutil.copytree(original_val, local_val)
        print("Done! You should now see hundreds of pictures in your VS Code 'datasets' folder.")

    train_dir = local_train
    validation_dir = local_val

    BATCH_SIZE = 32
    IMG_SIZE = (150, 150)

    print("Loading Training Images from memory...")
    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)

    print("Loading Validation Images from memory...")
    validation_dataset = image_dataset_from_directory(validation_dir,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)


    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


    def build_model(dropout_rate):
        model = Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(150, 150, 3)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    print("Beginning Hyper-Parameter Optimisation...")
    dropout_rates = [0.2, 0.5]
    best_model = None
    best_accuracy = 0
    results = {}

    for dropout in dropout_rates:
        print(f"\nTraining model with Dropout Rate: {dropout} (this will take a few minutes for higher accuracy)")
        model = build_model(dropout)
        

        history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset, verbose=1)

        loss, accuracy = model.evaluate(validation_dataset, verbose=0)
        print(f"Model Test Accuracy (Dropout {dropout}): {accuracy * 100:.2f}%")
        results[dropout] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"\nBest model achieved with Dropout {max(results, key=results.get)} at {best_accuracy*100:.2f}% accuracy.")
    best_model.save('animal_model.h5')
    print("Best model saved to 'animal_model.h5'")


    with open('hyperparameter_discussion.txt', 'w') as f:
        f.write("Hyper-Parameter Optimisation Results (Kaggle Dataset):\n")
        f.write("------------------------------------------------------\n")
        for dropout, acc in results.items():
            f.write(f"Dropout Rate {dropout}: Accuracy = {acc*100:.2f}%\n")
        
        best = max(results, key=results.get)
        f.write(f"\nConclusion: The optimal dropout rate was {best}.\n")
        f.write("Discussion:\n")
        f.write("Dropout helps prevent overfitting by randomly disabling neurons during training.\n")
        f.write(f"A dropout rate of {best} provided the best generalization to the unseen testing images from Kaggle.\n")


if __name__ == "__main__":
    train_and_save_model()
