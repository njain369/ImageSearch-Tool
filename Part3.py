import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine

# Load the pre-trained VGG16 models
model = keras.applications.VGG16(weights='imagenet', include_top=False, pooling='avg')


def calculateSimilarity(img1, img2):
    # Load and preprocess the input images
    image_path1 = img1
    image_path2 = img2
    img1 = image.load_img(image_path1, target_size=(224, 224))
    img2 = image.load_img(image_path2, target_size=(224, 224))
    x1 = image.img_to_array(img1)
    x2 = image.img_to_array(img2)
    x1 = preprocess_input(x1)
    x2 = preprocess_input(x2)
    
    # Extract features
    features1 = model.predict(tf.expand_dims(x1, axis=0)).flatten()
    features2 = model.predict(tf.expand_dims(x2, axis=0)).flatten()
    
    # Compute similarity using cosine similarity
    similarity = 1 - cosine(features1, features2)
    # Compute similarity percentage
    similarity_percentage = similarity * 100
    return similarity_percentage
