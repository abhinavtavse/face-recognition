import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from time import time
import os

classID = 0
outputFolderPath = 'data_collect_for_detect'
confidence = 0.90  # Set confidence threshold to 95%
maxBlurValue = 110  # Max allowed blur value
save = True
camWidth, camHeight = 640, 480
floatingPoint = 6

if not os.path.exists(outputFolderPath):
    os.makedirs(outputFolderPath)

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()

imageSaved = False  # Flag to check if the perfect image has been saved

while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []
    listInfo = []
    blurValue = 0  # Define blurValue outside the loop

    if bboxs and not imageSaved:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            if score >= confidence:  # Check if score is 95% or above
                # Ensure the face region is within valid image bounds
                if x >= 0 and y >= 0 and w > 0 and h > 0:
                    imgFace = img[y:y + h, x:x + w]
                    
                    # Check if the face region is non-empty and valid
                    if imgFace.size > 0:
                        blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                        if blurValue <= maxBlurValue:  # Check if blur value is 100 or less
                            # Conditions met: save the image
                            timeNow = str(time()).split('.')[0] + str(int(time() * 1000) % 1000)
                            imagePath = f"{outputFolderPath}/{timeNow}.jpg"
                            cv2.imwrite(imagePath, img)

                            labelFilePath = f"{outputFolderPath}/{timeNow}.txt"
                            with open(labelFilePath, 'a') as f:
                                # Save the face details as well
                                xc, yc = x + w / 2, y + h / 2
                                xcn, ycn = round(xc / img.shape[1], floatingPoint), round(yc / img.shape[0], floatingPoint)
                                wn, hn = round(w / img.shape[1], floatingPoint), round(h / img.shape[0], floatingPoint)
                                f.write(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                            imageSaved = True  # Set flag to true after saving the image
                            print("Desired image saved!")  # Display message

                            # Optional: exit the loop if only one image needs to be captured
                            break

            # Draw the face box and display score and blur value
            cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 20),
                               scale=2, thickness=3)

    cv2.imshow("Image", imgOut)
    if cv2.waitKey(1) & 0xFF == ord('q') or imageSaved:
        break

cap.release()
cv2.destroyAllWindows()
