import tensorflow as tf
from data_preprocessing import load_dataset, split_dataset
from model import DeepLabV3PlusDenseDDSSPP
from utils import CustomIoU, F1Score

# Paths
IMAGE_DIR = "/data/images"
MASK_DIR = "/data/masks"
MODEL_PATH = "./outputs/model_checkpoint.h5"

image_dataset, mask_dataset = load_dataset(IMAGE_DIR, MASK_DIR, size=512, max_images=max_images) # Enter the number of subdirectories as max_images

_, _, X_test, _, _, y_test = split_dataset(image_dataset, mask_dataset) # We are reusing this line do avoid downloading test_images


# Load the model
model = DeepLabV3PlusDenseDDSSPP(input_shape=(512, 512, 3))
model.load_weights(MODEL_PATH)

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', CustomIoU(num_classes=2, target_class_ids=[1]), F1Score()]
)

# Evaluate the model on the test set
test_loss, test_accuracy, test_iou, test_f1 = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test IoU: {test_iou}")
print(f"Test F1 Score: {test_f1}")
