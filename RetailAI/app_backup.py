from flask import Flask, render_template, jsonify, Response
import cv2
from ultralytics import YOLO
from collections import deque
from queue_monitor import analyze_queue

app = Flask(__name__)

# ==========================================
# YOLO MODEL
# ==========================================

model = YOLO("yolo11n.pt")

# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")

# ==========================================
# DETECTION SETTINGS
# ==========================================

CONFIDENCE = 0.60

HISTORY_SIZE = 10
MIN_AGREEMENT = 7

# ==========================================
# CUSTOMER COUNT
# ==========================================

customer_count = 0

count_history = deque(
    maxlen=HISTORY_SIZE
)

# ==========================================
# QUEUE DATA
# ==========================================

queue_count = 0

queue_status = "Normal"

queue_wait_time = 0

queue_alert = None


# ==========================================
# STABLE CUSTOMER COUNT
# ==========================================

def get_stable_count():

    global customer_count

    if len(count_history) == 0:
        return 0

    counts = list(count_history)

    frequency = {}

    for count in counts:

        if count not in frequency:
            frequency[count] = 0

        frequency[count] += 1

    most_common_count = max(
        frequency,
        key=frequency.get
    )

    most_common_frequency = frequency[
        most_common_count
    ]

    if most_common_frequency >= MIN_AGREEMENT:

        return most_common_count

    return customer_count


# ==========================================
# CAMERA FRAME GENERATOR
# ==========================================

def generate_frames():

    global customer_count
    global queue_count
    global queue_status
    global queue_wait_time
    global queue_alert

    while True:

        # ==================================
        # READ CAMERA
        # ==================================

        success, frame = camera.read()

        if not success:
            continue

        # ==================================
        # YOLO DETECTION
        # ==================================

        results = model(
            frame,
            classes=[0],
            conf=CONFIDENCE,
            verbose=False
        )

        detected_count = 0

        # ==================================
        # COUNT PEOPLE
        # ==================================

        if results and results[0].boxes:

            for box in results[0].boxes:

                class_id = int(
                    box.cls[0]
                )

                if class_id != 0:
                    continue

                confidence = float(
                    box.conf[0]
                )

                if confidence >= CONFIDENCE:

                    detected_count += 1

        # ==================================
        # STABILIZE CUSTOMER COUNT
        # ==================================

        count_history.append(
            detected_count
        )

        customer_count = get_stable_count()

        # ==================================
        # QUEUE MONITORING
        # ==================================
        #
        # TEMPORARY:
        #
        # For now we use the detected people
        # as the queue count.
        #
        # Later we will create a separate
        # billing-counter camera/queue zone.
        # ==================================

        queue_count = detected_count

        queue_result = analyze_queue(
            queue_count
        )

        queue_status = queue_result[
            "status"
        ]

        queue_wait_time = queue_result[
            "estimated_wait_minutes"
        ]

        queue_alert = queue_result[
            "alert"
        ]

        # ==================================
        # DRAW DETECTIONS
        # ==================================

        annotated_frame = results[0].plot()

        # ==================================
        # CUSTOMER COUNT
        # ==================================

        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (430, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated_frame,
            f"Customers Inside: {customer_count}",
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # ==================================
        # QUEUE COUNT
        # ==================================

        cv2.rectangle(
            annotated_frame,
            (10, 85),
            (430, 145),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated_frame,
            f"Queue: {queue_count}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        # ==================================
        # ENCODE FRAME
        # ==================================

        success, buffer = cv2.imencode(
            ".jpg",
            annotated_frame
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        # ==================================
        # SEND FRAME
        # ==================================

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/")
def dashboard():

    return render_template(
        "index.html"
    )


# ==========================================
# CUSTOMER + QUEUE API
# ==========================================

@app.route("/api/customer-count")
def get_customer_count():

    return jsonify({

        "customers": customer_count,

        "visitors": 0,

        "queue": queue_count,

        "queue_status": queue_status,

        "queue_wait": queue_wait_time,

        "queue_alert": queue_alert

    })


# ==========================================
# QUEUE API
# ==========================================

@app.route("/api/queue")
def get_queue():

    return jsonify({

        "people": queue_count,

        "status": queue_status,

        "estimated_wait_minutes":
            queue_wait_time,

        "alert": queue_alert

    })


# ==========================================
# HOURLY VISITORS
# ==========================================

@app.route("/api/hourly-visitors")
def hourly_visitors():

    return jsonify({})


# ==========================================
# VIDEO FEED
# ==========================================

@app.route("/video-feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print("       RetailAI Analytics System")
    print("======================================")
    print("")
    print("YOLO Person Detection: ACTIVE")
    print("Customer Count: ACTIVE")
    print("Queue Monitoring: ACTIVE")
    print("Total Visitors: DISABLED")
    print("")
    print("Dashboard:")
    print("http://127.0.0.1:5000")
    print("")
    print("Press CTRL+C to stop.")
    print("")

    app.run(
        debug=True,
        use_reloader=False
    )