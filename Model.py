import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Set paths to dataset folders (update these paths to your local directories)
NORMAL_DIR = r'C:\Users\asvat\Documents\Discovery\Model\Normal'
PNEUMONIA_DIR = r'C:\Users\asvat\Documents\Discovery\Model\Pneumonia'
TUBERCULOSIS_DIR = r'C:\Users\asvat\Documents\Discovery\Model\Tuberculosis'

# Function to create DataFrame from directory
def create_dataframe(directory, label):
    data = []
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        data.append([img_path, label])
    return pd.DataFrame(data, columns=['image_path', 'class'])

# Create DataFrames for each class
normal_df = create_dataframe(NORMAL_DIR, 'Normal')
pneumonia_df = create_dataframe(PNEUMONIA_DIR, 'Pneumonia')
tuberculosis_df = create_dataframe(TUBERCULOSIS_DIR, 'Tuberculosis')

# Combine datasets
df = pd.concat([normal_df, pneumonia_df, tuberculosis_df], ignore_index=True)

# Shuffle the combined dataset
df = df.sample(frac=1).reset_index(drop=True)

print(f"Total dataset size: {len(df)}")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])

# Function to display images
def display_images(images, titles, cols=5, fig_size=(15, 15)):
    rows = (len(images) - 1) // cols + 1
    fig, axs = plt.subplots(rows, cols, figsize=fig_size)
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axs[i // cols, i % cols] if rows > 1 else axs[i % cols]
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    for j in range(i + 1, rows * cols):
        axs[j // cols, j % cols].axis('off') if rows > 1 else axs[j % cols].axis('off')
    plt.tight_layout()
    plt.show()

# Display sample images from the dataset
def display_sample_images(df, num_samples=10):
    sample_df = df.sample(num_samples)
    images = []
    titles = []
    for _, row in sample_df.iterrows():
        img = load_img(row['image_path'], target_size=(224, 224))
        images.append(img)
        titles.append(f"Class: {row['class']}")
    display_images(images, titles)

print("Displaying sample images from the dataset:")
display_sample_images(df)

# Image preprocessing and data augmentation
def create_data_generators(data):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        preprocessing_function=applications.efficientnet.preprocess_input,
        validation_split=0.2
    )

    batch_size = 16
    img_size = (224, 224)

    train_generator = datagen.flow_from_dataframe(
        data,
        x_col='image_path',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_dataframe(
        data,
        x_col='image_path',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

# Define model
def build_model(num_classes):
    base_model = applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Compile model
def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Custom callback to print accuracy and detect overfitting
class MonitorOverfitting(keras.callbacks.Callback):
    def __init__(self, patience=5):
        super(MonitorOverfitting, self).__init__()
        self.patience = patience
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}")
        print(f"Training Accuracy: {logs['accuracy']:.4f}")
        print(f"Validation Accuracy: {logs['val_accuracy']:.4f}")
        print(f"Training Loss: {logs['loss']:.4f}")
        print(f"Validation Loss: {logs['val_loss']:.4f}")

        if logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            print(f"\nPossible overfitting detected. Stopping training.")
            self.model.stop_training = True

# Implement k-fold cross-validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

fold_scores = []

for fold, (train_index, val_index) in enumerate(kfold.split(df), 1):
    print(f"\nFold {fold}")

    train_data = df.iloc[train_index]
    val_data = df.iloc[val_index]

    train_generator, val_generator = create_data_generators(train_data)
    val_generator_fold = create_data_generators(val_data)[1]

    model = build_model(len(le.classes_))
    model = compile_model(model)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint(f'best_model_fold_{fold}.h5', save_best_only=True, monitor='val_loss'),
        MonitorOverfitting(patience=7)
    ]

    # Train model
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=False
    )

    # Evaluate model on validation set
    val_loss, val_acc = model.evaluate(val_generator_fold)
    print(f"\nValidation accuracy for fold {fold}: {val_acc:.4f}")
    fold_scores.append(val_acc)

    # Plot training history for this fold
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Print average score across all folds
print(f"\nAverage validation accuracy across all folds: {np.mean(fold_scores):.4f}")

# Select the best model
best_model = keras.models.load_model(f'best_model_fold_{np.argmax(fold_scores) + 1}.h5')

# Create a separate test set
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['class'], random_state=42)

# Create data generator for the test set
test_datagen = ImageDataGenerator(preprocessing_function=applications.efficientnet.preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Make predictions on the test set
predictions = best_model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Display some test images with their predictions
def display_test_images_with_predictions(test_df, true_classes, predicted_classes, num_samples=10):
    sample_indices = np.random.choice(len(test_df), num_samples, replace=False)
    images = []
    titles = []
    for idx in sample_indices:
        img = load_img(test_df.iloc[idx]['image_path'], target_size=(224, 224))
        images.append(img)
        true_label = le.inverse_transform([true_classes[idx]])[0]
        pred_label = le.inverse_transform([predicted_classes[idx]])[0]
        title = f"True: {true_label}\nPred: {pred_label}"
        titles.append(title)
    display_images(images, titles)

print("\nDisplaying test images with predictions:")
display_test_images_with_predictions(test_df, true_classes, predicted_classes)

# Classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=le.classes_))

# Calculate and print the final test accuracy
final_test_acc = np.mean(predicted_classes == true_classes)
print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Save the best model
best_model.save('lung_xray_model_kfold.h5')