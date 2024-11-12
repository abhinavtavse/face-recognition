import cv2
import os
import numpy as np
import shutil  # For deleting non-empty directories

# Paths
dataset_path = 'dataset_to_train'  # Folder where the training images are located
model_save_path = 'saved_model/s_model.yml'  # Path to save the trained model
live_capture_path = 'data_collect_from_live'  # Folder where live capture images are stored
detect_capture_path = 'data_collect_for_detect'  # Folder where detected images are stored

# Initialize the face detector and the LBPH face recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare to collect training data
faces = []
labels = []
label_map = {}  # To store the label corresponding to each person

# Load images and labels from dataset_to_train
for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)
    
    if image_path.endswith('.jpg'):  # Only process .jpg files
        img = cv2.imread(image_path)  # Read the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Detect faces in the image
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces_detected:
            # Crop the face region
            face = gray[y:y + h, x:x + w]

            # Assign a unique label (index) to each person
            label = image_name.split('.')[0]  # Assuming image_name is like 'person1.jpg'
            if label not in label_map:
                label_map[label] = len(label_map)
            
            # Store the face and corresponding label
            faces.append(face)
            labels.append(label_map[label])

# Train the model
recognizer.train(faces, np.array(labels))

# Save the trained model
if not os.path.exists('saved_model'):
    os.makedirs('saved_model')

recognizer.save(model_save_path)  # Save the trained model
print(f"Model trained and saved to {model_save_path}")

# Now, run the other scripts in order
os.system("python live_face_collect_to_compare.py")
os.system("python face_detect.py")
os.system("python face_recognition.py")

# Delete the saved model folder and its contents
if os.path.exists('saved_model'):
    shutil.rmtree('saved_model')  # Use shutil to remove non-empty directories
    print(f"Deleted {model_save_path}")

# Delete .jpg and .txt files in 'data_collect_from_live' and 'data_collect_for_detect' directories
def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.jpg') or filename.endswith('.txt'):
            os.remove(file_path)
            print(f"Deleted {file_path}")

delete_files_in_folder(live_capture_path)
delete_files_in_folder(detect_capture_path)

print("Finished processing and cleaning up!")
