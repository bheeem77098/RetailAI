# ==========================================
# RetailAI - Person Tracking Test
# ==========================================

from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open laptop camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("")
print("======================================")
print("       RetailAI Person Tracker")
print("======================================")
print("")
print("Person Tracking: ACTIVE")
print("Press Q to stop.")
print("")

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # YOLO detection + BoT-SORT tracking
    results = model.track(
        frame,
        classes=[0],
        conf=0.60,
        persist=True,
        tracker="trackers/retail_botsort.yaml",
        verbose=False
    )

    tracked_people = 0

    if results and results[0].boxes:

        boxes = results[0].boxes

        for box in boxes:

            # Only person class
            class_id = int(box.cls[0])

            if class_id != 0:
                continue

            tracked_people += 1

            # Get tracking ID
            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Display tracking ID
            if track_id is not None:

                label = f"Person ID: {track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # Display number of currently tracked people
    cv2.rectangle(
        frame,
        (10, 10),
        (300, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"People Tracked: {tracked_people}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show camera
    cv2.imshow(
        "RetailAI - Person Tracking",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("")
print("Person Tracking stopped.")
print("")