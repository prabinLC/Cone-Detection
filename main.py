import os
from inference import get_model
import supervision as sv
import cv2

# Set Roboflow API key
os.environ["ROBOFLOW_API_KEY"] = "rf_8CVUATKQ58fMli90K0zv5FD14AR2"

# load a pre-trained yolov8n model
model = get_model(model_id="traffic-cones-4laxg-xjuew/1")

# create supervision annotators with customization
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

# set confidence threshold (only show detections above this confidence)
CONFIDENCE_THRESHOLD = 0.7

# initialize webcam
cap = cv2.VideoCapture(0)

# check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Press 'q' to quit")

# main loop for real-time video processing
while True:
    # capture frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    try:
        # run inference on current frame
        results = model.infer(frame)[0]
        
        # load the results into the supervision Detections api
        detections = sv.Detections.from_inference(results)
        
        # filter detections by confidence threshold
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        
        # create labels with confidence scores
        labels = [
            f"cone: {confidence:.2f}"
            for confidence in detections.confidence
        ]
        
        # annotate the frame with detection results
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)
        
        # add info text about detection count and threshold
        detection_count = len(detections)
        cv2.putText(annotated_frame, f"Cones detected: {detection_count} (conf > {CONFIDENCE_THRESHOLD})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Press 'q' to quit, '+'/'-' to adjust threshold", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # display the annotated frame
        cv2.imshow('Traffic Cone Detection', annotated_frame)
        
        # handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            CONFIDENCE_THRESHOLD = min(0.95, CONFIDENCE_THRESHOLD + 0.05)
            print(f"Confidence threshold increased to: {CONFIDENCE_THRESHOLD:.2f}")
        elif key == ord('-'):
            CONFIDENCE_THRESHOLD = max(0.05, CONFIDENCE_THRESHOLD - 0.05)
            print(f"Confidence threshold decreased to: {CONFIDENCE_THRESHOLD:.2f}")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        # show original frame if inference fails
        cv2.imshow('Traffic Cone Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed")