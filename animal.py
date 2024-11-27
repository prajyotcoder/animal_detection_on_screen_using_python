import cv2
import numpy as np
import pyautogui
import time

# Load YOLO model (yolov3.weights, yolov3.cfg)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the class labels YOLO is trained on (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Filter classes related to animals (e.g., dog, cat, cow, etc.)
animal_classes = ["dog", "cat", "cow", "horse", "sheep"]

# Function to monitor the screen and detect animals
def monitor_screen():
    # Create a named window only once with the ability to resize
    window_name = "Animal Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set window size once (e.g., 800x600)
    cv2.resizeWindow(window_name, 800, 600)

    # Set to track detected animals (to avoid counting duplicates)
    detected_animals = set()

    # Animal count
    total_count = 0

    while True:
        # Capture a screenshot of the screen
        screenshot = pyautogui.screenshot()

        # Convert the screenshot to a numpy array
        frame = np.array(screenshot)

        # Convert RGB (Pillow) to BGR (OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Get the dimensions of the frame
        height, width, channels = frame.shape

        # Prepare the frame for input to YOLO (resize and normalize)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Set the input for the YOLO network
        net.setInput(blob)

        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Get predictions
        detections = net.forward(output_layers)

        # Reset count for new frame
        frame_animal_count = 0

        # Flag to detect any animal
        animal_detected = False

        # Process detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] in animal_classes:
                    # Get the bounding box coordinates
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Use the bounding box as a unique identifier to avoid counting the same animal
                    bbox_id = (x, y, w, h)

                    # Check if the animal has been detected already in this frame
                    if bbox_id not in detected_animals:
                        # Add to the set of detected animals
                        detected_animals.add(bbox_id)
                        frame_animal_count += 1  # Increment animal count for new detection

                        # Draw bounding box and label
                        label = f"{classes[class_id]}: {confidence * 100:.2f}%"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

                        # Set flag to true as soon as an animal is detected
                        animal_detected = True

        # If animal is detected, clear the frame and reset the detection
        if animal_detected:
            # Display the frame with bounding boxes and animal count
            total_count += frame_animal_count
            count_label = f"Total Animals Detected: {total_count}"
            cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display the frame
            cv2.imshow(window_name, frame)

            # Clear the detected animals set to start fresh
            detected_animals.clear()

            # Wait for 1 second to allow detection to be visible
            time.sleep(1)

            # Restart the process after animal is detected (clear screen)
            continue

        # If no animal detected, continue showing the frame
        else:
            # Display the animal count on the screen
            count_label = f"Total Animals Detected: {total_count}"
            cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display the frame with bounding boxes and animal count
            cv2.imshow(window_name, frame)

        # Exit condition (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the window after the loop
    cv2.destroyAllWindows()

# Start the screen monitoring function
monitor_screen()
