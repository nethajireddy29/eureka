import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing, applications, regularizers, optimizers
import matplotlib.pyplot as plt

train_datagen = preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rotation_range = 20,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

val_datagen = preprocessing.image.ImageDataGenerator(rescale = 1./255)
test_datagen = preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_genertor = train_datagen.flow_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/train',
    target_size = (244, 244),
    batch_size = 32,
    class_mode = 'binary'
)
val_genertor = val_datagen.flow_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/val',
    target_size = (244, 244),
    batch_size = 32,
    class_mode = 'binary'
)
test_genertor = test_datagen.flow_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/test',
    target_size = (244, 244),
    batch_size = 32,
    class_mode = 'binary'
)

base_model = applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(244, 244, 3)
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') 
])

model.compile(
    optimizer = optimizers.Adam(learning_rate = 0.001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(
    train_genertor,
    epochs = 5,
    validation_data = val_genertor
)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

test_loss, test_accuracy = model.evaluate(test_genertor)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

def analyze_fit(history):
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    acc_diff = training_accuracy[-1] - validation_accuracy[-1]
    loss_diff = validation_loss[-1] - training_loss[-1]

    print(f"Final Training Accuracy: {training_accuracy[-1]:.4f}")
    print(f"Final Validation Accuracy: {validation_accuracy[-1]:.4f}")
    print(f"Accuracy Difference (Training - Validation): {acc_diff:.4f}\n")
    print(f"Final Training Loss: {training_loss[-1]:.4f}")
    print(f"Final Validation Loss: {validation_loss[-1]:.4f}")
    print(f"Loss Difference (Validation - Training): {loss_diff:.4f}\n")


    if acc_diff > 0.1 and loss_diff > 0.1: 
        print("The model is likely overfitting. Try techniques like regularization or dropout.")
    elif acc_diff < -0.1:
        print("The model might be underfitting. Consider increasing model capacity or training longer.")
    else:
        print("The model seems to have a good fit. Continue fine-tuning for optimal performance.")

analyze_fit(history)