import cv2
from ultralytics import YOLO


# Load YOLO model
model = YOLO("yolo11n.pt")


# Open laptop camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()


print("AI Person Detection Started")
print("Press Q to stop.")


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break


    # Run YOLO detection
    results = model(frame, verbose=False)


    # Count people
    people_count = 0


    for result in results:

        boxes = result.boxes

        for box in boxes:

            # COCO class 0 = person
            class_id = int(box.cls[0])

            if class_id == 0:
                people_count += 1


    # Draw YOLO detections
    annotated_frame = results[0].plot()


    # Display customer count
    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (300, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated_frame,
        f"Customers Detected: {people_count}",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # Show result
    cv2.imshow(
        "RetailAI - AI Person Detection",
        annotated_frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release camera
camera.release()
cv2.destroyAllWindows()

print("AI Detection Stopped.")