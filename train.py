import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from data_preprocessing import load_dataset, split_dataset
from model import DeepLabV3PlusDenseDDSSPP
from utils import CustomIoU, F1Score
import os

IMAGE_DIR = "/data/images"
MASK_DIR = "/data/masks"
MODEL_SAVE_PATH = "outputs/model_checkpoint.h5"


image_dataset, mask_dataset = load_dataset(IMAGE_DIR, MASK_DIR, size=512, max_images=max_images) # Enter the number of subdirectories as max_images


X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(image_dataset, mask_dataset)

model = DeepLabV3PlusDenseDDSSPP(input_shape=(512, 512, 3))


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.96, staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', CustomIoU(num_classes=2, target_class_ids=[1]), F1Score()]
)


checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[checkpoint],
    verbose=1
)
