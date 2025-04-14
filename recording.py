import cv2
import cvzone
import time
import datetime
from ultralytics import YOLO

print("OpenCV version:", cv2.__version__)

print()
yolo11 = YOLO("persondetect.pt")

cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

#for recording
recording=False
start_counter=0
end_counter=0
start_thread=15
end_thread=15
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640,  480))

while True:
    frameCaptured, frame = cap.read()

    if not frameCaptured:
        break
    results = yolo11(frame)

    # Get confidence scores and filter boxes
    boxes = results[0].boxes
    high_conf_boxes = [box for box in boxes if box.conf[0] > 0.7]  # Filter boxes with confidence > 0.7

    # Create a new frame for plotting
    processedFrame = frame.copy()

    # Draw bounding boxes for high confidence detections
    for box in high_conf_boxes:
        xyxy = box.xyxy[0]  # Get the bounding box coordinates
        cv2.rectangle(processedFrame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(processedFrame, f'{box.conf[0]:.2f}', (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    high_confidence_detected = len(high_conf_boxes) > 0  # Check if any high confidence boxes exist

    if high_confidence_detected:
        end_counter =0
        if start_counter > start_thread:
            if not recording:
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out = cv2.VideoWriter(f'{current_time}.avi', fourcc, 10.0, (640,  480))
                test=test+1
                recording = True
            record_frame = cv2.flip(processedFrame, 0)
            out.write(frame)
            
        else:
            start_counter = start_counter+1
        
    else:

        if end_counter >end_thread:
            out.release
            recording=False
            end_counter=0
            start_counter=0
        else:
            end_counter=end_counter+1
        
    cv2.putText(processedFrame, f'Start Counter: {start_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(processedFrame, f'End Counter: {end_counter}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(processedFrame, f'state: {recording}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    xyxy=results[0].boxes.xyxy
    print(xyxy)

    cv2.imshow("YOLO Live Webcam", processedFrame)

    if cv2.waitKey(1) & 0xFF == 27:
        break