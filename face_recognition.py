import cv2
import os
from cv2 import face

# Paths
data_collect_folder = 'data_collect_from_live'
uploaded_image_path = 'data_collect_for_detect'
model_path = 'saved_model/s_model.yml'

# Load the first image file from data_collect
image1_path = None
for file in os.listdir(data_collect_folder):
    if file.endswith('.jpg'):
        image1_path = os.path.join(data_collect_folder, file)
        break

# Check if an image was found in data_collect
if image1_path is None:
    print("No .jpg file found in data_collect.")
    exit()

# Load the uploaded image and model
image2_path = None
for file in os.listdir(uploaded_image_path):
    if file.endswith('.jpg'):
        image2_path = os.path.join(uploaded_image_path, file)
        break

# Check if an image was found in data_collect
if image2_path is None:
    print("No .jpg file found in data_collect.")
    exit()

# Function to extract face features
def get_face(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path {img_path} not found.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError(f"No face detected in image {img_path}.")

    x, y, w, h = faces[0]  # Assuming the first detected face is the target
    return gray[y:y+h, x:x+w]

# Load model
recognizer = face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Get face regions for comparison
face1 = get_face(image1_path)
face2 = get_face(image2_path)

# Perform comparison and get similarity score
label, confidence = recognizer.predict(face1)
similarity_score = max(0, (100 - confidence)+30)  # Higher score = more similar

print(f"Similarity score between images: {similarity_score:.2f}%")
