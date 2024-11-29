import tensorflow as tf
print("TensorFlow Version:", tf.__version__)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)
print("TensorFlow Object Detection API is installed!")