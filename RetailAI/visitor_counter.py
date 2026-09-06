# ==========================================
# RetailAI - Entry / Exit Visitor Counter
# ==========================================

from ultralytics import YOLO
import cv2

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.60

# Virtual counting line
LINE_Y = 300

# -----------------------------
# Load YOLO model
# -----------------------------

model = YOLO(MODEL_PATH)

# -----------------------------
# Open camera
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

# -----------------------------
# Visitor counters
# -----------------------------

total_entries = 0
total_exits = 0

# Stores previous center Y position for each tracked person
previous_positions = {}

print("")
print("======================================")
print("       RetailAI Visitor Counter")
print("======================================")
print("")
print("Person Detection : ACTIVE")
print("Person Tracking  : ACTIVE")
print("Entry Detection  : ACTIVE")
print("Exit Detection   : ACTIVE")
print("")
print("Virtual Line Position:", LINE_Y)
print("")
print("Move across the line to test.")
print("Press Q to stop.")
print("")

# -----------------------------
# Main loop
# -----------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # -----------------------------
    # Track people
    # -----------------------------

    results = model.track(
        frame,
        classes=[0],
        conf=CONFIDENCE,
        persist=True,
        tracker="trackers/retail_botsort.yaml",
        verbose=False
    )

    # -----------------------------
    # Draw virtual counting line
    # -----------------------------

    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (255, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "ENTRY / EXIT LINE",
        (20, LINE_Y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # Process tracked people
    # -----------------------------

    if results and results[0].boxes:

        boxes = results[0].boxes

        for box in boxes:

            class_id = int(box.cls[0])

            # Only person class
            if class_id != 0:
                continue

            # Need tracking ID
            if box.id is None:
                continue

            track_id = int(box.id[0])

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Person center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 255, 255),
                -1
            )

            # Show ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # -----------------------------
            # Check previous position
            # -----------------------------

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                # -----------------------------
                # Moving DOWN across line
                # -----------------------------

                if (
                    previous_y < LINE_Y
                    and center_y >= LINE_Y
                ):

                    total_entries += 1

                    print(
                        f"ENTRY detected - Person ID: {track_id}"
                    )

                # -----------------------------
                # Moving UP across line
                # -----------------------------

                elif (
                    previous_y > LINE_Y
                    and center_y <= LINE_Y
                ):

                    total_exits += 1

                    print(
                        f"EXIT detected - Person ID: {track_id}"
                    )

            # Save current position
            previous_positions[track_id] = center_y

    # -----------------------------
    # Customers currently inside
    # -----------------------------

    customers_inside = total_entries - total_exits

    if customers_inside < 0:
        customers_inside = 0

    # -----------------------------
    # Information panel
    # -----------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (400, 145),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Total Visitors: {total_entries}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Entries: {total_entries}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Exits: {total_exits}",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Inside Now: {customers_inside}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # Display
    # -----------------------------

    cv2.imshow(
        "RetailAI - Visitor Counter",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------

camera.release()
cv2.destroyAllWindows()

print("")
print("======================================")
print("       Visitor Counter Stopped")
print("======================================")
print("")
print("Total Visitors:", total_entries)
print("Total Entries :", total_entries)
print("Total Exits   :", total_exits)
print(
    "Customers Inside:",
    max(0, total_entries - total_exits)
)
print("")