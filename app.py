from flask import Flask, render_template, Response, jsonify
import cv2
import time
import threading
from collections import deque, Counter
from ultralytics import YOLO

from zone_analytics import update_zone_analytics
from queue_monitor import analyze_queue
from database import get_db_connection


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_PATH = "yolo11n.pt"

model = YOLO(MODEL_PATH)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# AI SETTINGS
# ============================================================

CONFIDENCE = 0.50

HISTORY_SIZE = 5
MIN_AGREEMENT = 3

DATABASE_SAVE_INTERVAL = 30


# ============================================================
# GLOBAL DATA
# ============================================================

customer_count = 0
tracked_people = 0

latest_frame = None

data_lock = threading.Lock()


# ============================================================
# CUSTOMER COUNT STABILIZATION
# ============================================================

count_history = deque(maxlen=HISTORY_SIZE)


def get_stable_customer_count(new_count):

    global customer_count

    count_history.append(new_count)

    if len(count_history) < HISTORY_SIZE:
        return customer_count

    counter = Counter(count_history)

    most_common_count, frequency = counter.most_common(1)[0]

    if frequency >= MIN_AGREEMENT:

        customer_count = most_common_count

    return customer_count


# ============================================================
# CUSTOMER OBSERVATION / DWELL ANALYTICS
# ============================================================

# Stores the first and last time each tracking ID
# was visible during the current application session.

person_first_seen = {}
person_last_seen = {}

observation_lock = threading.Lock()


def update_observation_times(track_ids):

    current_time = time.time()

    with observation_lock:

        for track_id in track_ids:

            if track_id not in person_first_seen:

                person_first_seen[track_id] = current_time

            person_last_seen[track_id] = current_time


def get_average_observation_time():

    current_time = time.time()

    durations = []

    with observation_lock:

        for track_id in person_first_seen:

            first_seen = person_first_seen[track_id]

            last_seen = person_last_seen.get(
                track_id,
                current_time
            )

            duration = last_seen - first_seen

            if duration >= 0:

                durations.append(duration)


    if not durations:

        return 0


    return round(
        sum(durations) / len(durations),
        1
    )


# ============================================================
# HOURLY CUSTOMER ACTIVITY
# ============================================================

hourly_activity = {}

hourly_lock = threading.Lock()


def update_hourly_activity(customer_value):

    current_hour = time.strftime("%H:00")

    with hourly_lock:

        if current_hour not in hourly_activity:

            hourly_activity[current_hour] = []

        hourly_activity[current_hour].append(
            customer_value
        )


def get_peak_hour():

    with hourly_lock:

        if not hourly_activity:

            return None

        hour_averages = {}

        for hour, values in hourly_activity.items():

            if values:

                hour_averages[hour] = (
                    sum(values) / len(values)
                )


        if not hour_averages:

            return None


        peak_hour = max(
            hour_averages,
            key=hour_averages.get
        )


        peak_value = round(
            hour_averages[peak_hour]
        )


        return {
            "hour": peak_hour,
            "average_customers": peak_value
        }


# ============================================================
# ZONE DEFINITIONS
# ============================================================

ZONE_NAMES = [
    "Beverages",
    "Snacks",
    "Fruits & Veg",
    "Dairy",
    "Home Care"
]


# ============================================================
# CURRENT ZONE COUNTS
# ============================================================

zone_counts = {

    "Beverages": 0,

    "Snacks": 0,

    "Fruits & Veg": 0,

    "Dairy": 0,

    "Home Care": 0

}


# ============================================================
# ZONE VISIT COUNTS
# ============================================================

zone_visits = {

    "Beverages": 0,

    "Snacks": 0,

    "Fruits & Veg": 0,

    "Dairy": 0,

    "Home Care": 0

}


# ============================================================
# QUEUE DATA
# ============================================================

queue_data = {

    "people": 0,

    "status": "Normal",

    "estimated_wait_minutes": 0,

    "alert": None

}


# ============================================================
# DATABASE SAVE
# ============================================================

last_database_save = 0


def save_to_database():

    global last_database_save

    current_time = time.time()

    if (
        current_time - last_database_save
        < DATABASE_SAVE_INTERVAL
    ):

        return


    last_database_save = current_time


    try:

        connection = get_db_connection()

        cursor = connection.cursor()


        # -----------------------------------------
        # CUSTOMER ANALYTICS
        # -----------------------------------------

        cursor.execute(

            """
            INSERT INTO customer_analytics
            (customer_count)
            VALUES (%s)
            """,

            (customer_count,)

        )


        # -----------------------------------------
        # ZONE ANALYTICS
        # -----------------------------------------

        for zone in ZONE_NAMES:

            cursor.execute(

                """
                INSERT INTO zone_analytics
                (zone_name, current_count, visits)
                VALUES (%s, %s, %s)
                """,

                (
                    zone,
                    zone_counts.get(
                        zone,
                        0
                    ),
                    zone_visits.get(
                        zone,
                        0
                    )
                )

            )


        # -----------------------------------------
        # QUEUE ANALYTICS
        # -----------------------------------------

        cursor.execute(

            """
            INSERT INTO queue_analytics
            (people, status, estimated_wait_minutes)
            VALUES (%s, %s, %s)
            """,

            (
                queue_data.get(
                    "people",
                    0
                ),

                queue_data.get(
                    "status",
                    "Normal"
                ),

                queue_data.get(
                    "estimated_wait_minutes",
                    0
                )

            )

        )


        # -----------------------------------------
        # HOURLY ANALYTICS
        # -----------------------------------------

        current_hour = time.strftime(
            "%H:00"
        )


        cursor.execute(

            """
            INSERT INTO hourly_analytics
            (hour_label, visitor_count)
            VALUES (%s, %s)
            """,

            (
                current_hour,
                customer_count
            )

        )


        connection.commit()

        cursor.close()

        connection.close()


        print(
            "Database updated successfully."
        )


    except Exception as error:

        print(
            "Database save error:",
            error
        )


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global latest_frame
    global tracked_people
    global zone_counts
    global zone_visits
    global queue_data


    while True:

        success, frame = camera.read()


        if not success:

            print(
                "Unable to read camera."
            )

            time.sleep(1)

            continue


        try:

            # -----------------------------------------
            # YOLO + BOT-SORT TRACKING
            # -----------------------------------------

            results = model.track(

                frame,

                classes=[0],

                conf=CONFIDENCE,

                persist=True,

                tracker="trackers/retail_botsort.yaml",

                verbose=False

            )


            detected_people = 0

            person_boxes = []

            current_track_ids = []


            # -----------------------------------------
            # PROCESS DETECTIONS
            # -----------------------------------------

            if results and len(results) > 0:

                result = results[0]


                if result.boxes is not None:

                    boxes = result.boxes


                    for box in boxes:

                        detected_people += 1


                        coordinates = (

                            box.xyxy[0]

                            .cpu()

                            .numpy()

                            .astype(int)

                        )


                        x1, y1, x2, y2 = coordinates


                        person_boxes.append(

                            (
                                x1,
                                y1,
                                x2,
                                y2
                            )

                        )


                        # ---------------------------------
                        # GET BOT-SORT TRACK ID
                        # ---------------------------------

                        if box.id is not None:

                            track_id = int(
                                box.id[0]
                                .cpu()
                                .item()
                            )

                            current_track_ids.append(
                                track_id
                            )


            # -----------------------------------------
            # OBSERVATION TIME
            # -----------------------------------------

            if current_track_ids:

                update_observation_times(
                    current_track_ids
                )


            # -----------------------------------------
            # STABLE CUSTOMER COUNT
            # -----------------------------------------

            stable_count = (
                get_stable_customer_count(
                    detected_people
                )
            )


            tracked_people = stable_count


            # -----------------------------------------
            # HOURLY ACTIVITY
            # -----------------------------------------

            update_hourly_activity(
                stable_count
            )


            # -----------------------------------------
            # ZONE ANALYTICS
            # -----------------------------------------

            try:

                zone_result = (
                    update_zone_analytics(
                        frame,
                        person_boxes
                    )
                )


                if isinstance(
                    zone_result,
                    tuple
                ):

                    current_zones = (
                        zone_result[0]
                    )

                    visits_result = (

                        zone_result[1]

                        if len(zone_result) > 1

                        else None

                    )

                else:

                    current_zones = (
                        zone_result
                    )

                    visits_result = None


                if isinstance(
                    current_zones,
                    dict
                ):

                    for zone in ZONE_NAMES:

                        if zone in current_zones:

                            value = (
                                current_zones[zone]
                            )


                            # ---------------------------------
                            # Dictionary format
                            # ---------------------------------

                            if isinstance(
                                value,
                                dict
                            ):

                                zone_counts[zone] = int(

                                    value.get(
                                        "current",
                                        0
                                    )

                                )


                                if "visits" in value:

                                    zone_visits[zone] = int(

                                        value.get(
                                            "visits",
                                            0
                                        )

                                    )


                            # ---------------------------------
                            # Integer format
                            # ---------------------------------

                            else:

                                zone_counts[zone] = int(
                                    value
                                )


                if isinstance(
                    visits_result,
                    dict
                ):

                    for zone in ZONE_NAMES:

                        if zone in visits_result:

                            zone_visits[zone] = int(

                                visits_result[zone]

                            )


            except Exception as error:

                print(
                    "Zone analytics error:",
                    error
                )


            # -----------------------------------------
            # QUEUE ANALYTICS
            # -----------------------------------------
            #
            # Prototype only.
            #
            # This currently uses stabilized
            # customer detection.
            #
            # It is NOT dedicated billing-counter
            # queue detection.
            #

            try:

                queue_data = (
                    analyze_queue(
                        stable_count
                    )
                )


            except Exception as error:

                print(
                    "Queue analytics error:",
                    error
                )


                queue_data = {

                    "people":
                        stable_count,

                    "status":
                        "Normal",

                    "estimated_wait_minutes":
                        stable_count * 2,

                    "alert":
                        None

                }


            # -----------------------------------------
            # DRAW PERSON BOXES
            # -----------------------------------------

            for (

                x1,
                y1,
                x2,
                y2

            ) in person_boxes:


                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2

                )


                cv2.putText(

                    frame,

                    "Person",

                    (x1, y1 - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 0),

                    2

                )


            # -----------------------------------------
            # CAMERA OVERLAY
            # -----------------------------------------

            cv2.rectangle(

                frame,

                (15, 15),

                (350, 90),

                (0, 0, 0),

                -1

            )


            cv2.putText(

                frame,

                f"Customers Inside: {stable_count}",

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2

            )


            cv2.putText(

                frame,

                "RetailAI - AI Monitoring",

                (30, 78),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                1

            )


            # -----------------------------------------
            # SAVE FRAME
            # -----------------------------------------

            with data_lock:

                latest_frame = frame.copy()


            # -----------------------------------------
            # DATABASE
            # -----------------------------------------

            save_to_database()


        except Exception as error:

            print(
                "Camera processing error:",
                error
            )


            with data_lock:

                latest_frame = frame.copy()


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    while True:

        with data_lock:

            if latest_frame is None:

                continue

            frame = latest_frame.copy()


        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )


        if not success:

            continue


        frame_bytes = buffer.tobytes()


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"

        )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    return render_template(
        "index.html"
    )


# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
def reports():

    return render_template(
        "reports.html"
    )


# ============================================================
# LIVE VIDEO
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=
        "multipart/x-mixed-replace; boundary=frame"

    )


# ============================================================
# LIVE DATA API
# ============================================================

@app.route("/api/live-data")
def live_data():

    zones = {}


    for zone in ZONE_NAMES:

        zones[zone] = {

            "current":
                zone_counts.get(
                    zone,
                    0
                ),

            "visits":
                zone_visits.get(
                    zone,
                    0
                )

        }


    peak_hour = get_peak_hour()


    return jsonify({

        "customers_inside":
            customer_count,

        "tracked_people":
            tracked_people,

        "zone_visits":
            zone_visits,

        "zones":
            zones,

        "queue":
            queue_data,

        "average_observation_time":
            get_average_observation_time(),

        "peak_hour":
            peak_hour

    })


# ============================================================
# CUSTOMER ANALYTICS API
# ============================================================

@app.route("/api/customer-analytics")
def customer_analytics():

    average_time = (
        get_average_observation_time()
    )

    peak_hour = get_peak_hour()


    # Convert seconds into minutes
    average_minutes = round(
        average_time / 60,
        1
    )


    return jsonify({

        "customers_inside":
            customer_count,

        "tracked_people":
            tracked_people,

        "average_observation_seconds":
            average_time,

        "average_observation_minutes":
            average_minutes,

        "peak_hour":
            peak_hour

    })


# ============================================================
# HOURLY VISITOR API
# ============================================================

@app.route("/api/hourly-visitors")
def hourly_visitors():

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )


        cursor.execute(

            """
            SELECT

                DATE_FORMAT(
                    MIN(recorded_at),
                    '%H:00'
                ) AS hour_label,

                ROUND(
                    AVG(customer_count),
                    0
                ) AS customer_traffic

            FROM customer_analytics

            WHERE recorded_at >=
                NOW() - INTERVAL 24 HOUR

            GROUP BY
                HOUR(recorded_at)

            ORDER BY
                HOUR(recorded_at)

            """

        )


        rows = cursor.fetchall()


        cursor.close()

        connection.close()


        result = {}


        for row in rows:

            result[
                row["hour_label"]
            ] = int(

                row["customer_traffic"]
                or 0

            )


        return jsonify(result)


    except Exception as error:

        print(
            "Hourly analytics error:",
            error
        )


        return jsonify({})


# ============================================================
# PEAK HOUR API
# ============================================================

@app.route("/api/peak-hour")
def peak_hour_api():

    try:

        connection = get_db_connection()

        cursor = connection.cursor(
            dictionary=True
        )


        cursor.execute(

            """
            SELECT

                DATE_FORMAT(
                    MIN(recorded_at),
                    '%H:00'
                ) AS hour_label,

                ROUND(
                    AVG(customer_count),
                    0
                ) AS average_customers

            FROM customer_analytics

            WHERE recorded_at >=
                NOW() - INTERVAL 24 HOUR

            GROUP BY
                HOUR(recorded_at)

            ORDER BY
                average_customers DESC

            LIMIT 1

            """

        )


        row = cursor.fetchone()


        cursor.close()

        connection.close()


        if row is None:

            return jsonify({

                "hour":
                    None,

                "average_customers":
                    0

            })


        return jsonify({

            "hour":
                row["hour_label"],

            "average_customers":
                int(
                    row[
                        "average_customers"
                    ] or 0
                )

        })


    except Exception as error:

        print(
            "Peak hour error:",
            error
        )


        return jsonify({

            "hour":
                None,

            "average_customers":
                0

        })


# ============================================================
# ZONE ANALYTICS API
# ============================================================

@app.route("/api/zones")
def zones_api():

    zones = {}


    for zone in ZONE_NAMES:

        zones[zone] = {

            "current":
                zone_counts.get(
                    zone,
                    0
                ),

            "visits":
                zone_visits.get(
                    zone,
                    0
                )

        }


    return jsonify(zones)


# ============================================================
# QUEUE API
# ============================================================

@app.route("/api/queue")
def queue_api():

    return jsonify(
        queue_data
    )


# ============================================================
# DATABASE STATUS
# ============================================================

@app.route("/api/database-status")
def database_status():

    try:

        connection = get_db_connection()

        cursor = connection.cursor()


        cursor.execute(
            "SELECT 1"
        )


        cursor.fetchone()


        cursor.close()

        connection.close()


        return jsonify({

            "status":
                "connected",

            "database":
                "retailai"

        })


    except Exception as error:

        print(
            "Database status error:",
            error
        )


        return jsonify({

            "status":
                "disconnected",

            "database":
                "retailai"

        })


# ============================================================
# START CAMERA THREAD
# ============================================================

camera_thread = threading.Thread(

    target=camera_loop,

    daemon=True

)

camera_thread.start()


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=False,

        host="127.0.0.1",

        port=5000,

        use_reloader=False

    )