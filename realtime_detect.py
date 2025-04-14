import cv2
import cvzone
import time
from ultralytics import YOLO

print("OpenCV version:", cv2.__version__)
# Loading both models
#yolo10 = YOLO("yolov10n.pt")
yolo11 = YOLO(r"C:\Users\user\Desktop\FYP\fyp resource\550+CT GSConv\GSConv550+CT.pt")


# Lists to store nums for respective measurements
inferenceList = []
fpsList = []
confidenceList = []

# Capturing live webcam footage
cap = cv2.VideoCapture(0)

# Inf loop to get all the frames
while True:
    frameCaptured, frame = cap.read()

    if not frameCaptured:
        break

    # Getting time value of which the inferencing begins
    startTime = time.time()

    results = yolo11(frame)

    # Plotting all objects found on frame without customising any of the visuals.
    processedFrame = results[0].plot()

    # Calculating inference time and converting to FPS
    inferenceTime = time.time() - startTime
    fps = 1 / inferenceTime

    # Adding values to a list
    inferenceList.append(inferenceTime)
    fpsList.append(fps)

    # Get confidence scores for all detected objects
    confidences = [box.conf[0] * 100 for box in results[0].boxes]
    if confidences:
        avgConfidence = sum(confidences) / len(confidences)
    else:

        # Handle case where no objects are detected
        avgConfidence = 0.0  

    # Adding to a list
    #confidenceList.append(avgConfidence)

    
    #cvzone.putTextRect(processedFrame, f"FPS: {fps:.2f}", (10, 30), 2)
    #cvzone.putTextRect(processedFrame, f"Inference Time: {inferenceTime:.4f} s", (10, 70), 2)

    xyxy=results[0].boxes.xyxy
    print(xyxy)

    cv2.imshow("YOLO Live Webcam", processedFrame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Getting average value of all the metrics
#avgFPS = sum(fpsList) / len(fpsList)
#avgInference = sum(inferenceList) / len(inferenceList)
#avgConfidence = sum(confidenceList) / len(confidenceList)

# Printing values to console.
#print(f"Average FPS: {avgFPS:.2f}")
#print(f"Average Inference Time: {avgInference:.4f}")
#print(f"Average Confidence: {avgConfidence:.2f}")

# YOLO 10 results
# FPS: 13.04
# INFERENCE: 0.0796s
# CONF: 81.95%

# YOLO 11 RESULTS
# FPS: 19.10
# Inference: 0.0546s
# Conf: 70.05%