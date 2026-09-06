# ==========================================
# RetailAI - Store Area Analytics Module
# ==========================================

from ultralytics import YOLO


# ------------------------------------------
# Configuration
# ------------------------------------------

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.60

ZONE_NAMES = [
    "Beverages",
    "Snacks",
    "Fruits & Veg",
    "Dairy",
    "Home Care"
]


# ------------------------------------------
# Zone Visit Counters
# ------------------------------------------

zone_visits = {
    name: 0
    for name in ZONE_NAMES
}


# ------------------------------------------
# Track which zones each person visited
# ------------------------------------------

person_zone_history = {}


# ------------------------------------------
# Load YOLO model
# ------------------------------------------
#
# IMPORTANT:
# This module does NOT open a camera.
# The main app.py owns the camera.
#

model = YOLO(MODEL_PATH)


# ------------------------------------------
# Main Zone Analytics Function
# ------------------------------------------

def update_zone_analytics(frame, person_boxes):

    """
    Analyze detected people and divide the
    camera frame into 5 vertical store zones.

    person_boxes:
        List of tuples:
        (x1, y1, x2, y2)

    Returns:
        current_zone_counts,
        zone_visits
    """

    frame_height, frame_width = frame.shape[:2]


    # --------------------------------------
    # Create 5 vertical zones
    # --------------------------------------

    zone_width = (
        frame_width //
        len(ZONE_NAMES)
    )


    zones = []


    for i, name in enumerate(ZONE_NAMES):

        x1 = i * zone_width


        if i == len(ZONE_NAMES) - 1:

            x2 = frame_width

        else:

            x2 = (
                (i + 1) *
                zone_width
            )


        zones.append(
            {
                "name": name,
                "x1": x1,
                "y1": 0,
                "x2": x2,
                "y2": frame_height
            }
        )


    # --------------------------------------
    # Current people in each zone
    # --------------------------------------

    current_zone_counts = {
        name: 0
        for name in ZONE_NAMES
    }


    # --------------------------------------
    # Process detected people
    # --------------------------------------

    for index, box in enumerate(person_boxes):

        x1, y1, x2, y2 = box


        # ----------------------------------
        # Person center
        # ----------------------------------

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )


        # ----------------------------------
        # Use detection index as temporary ID
        #
        # Main app already handles tracking.
        # ----------------------------------

        track_id = index


        current_zone = None


        # ----------------------------------
        # Find current zone
        # ----------------------------------

        for zone in zones:

            if (

                zone["x1"]
                <= center_x
                < zone["x2"]

                and

                zone["y1"]
                <= center_y
                < zone["y2"]

            ):

                current_zone = (
                    zone["name"]
                )

                break


        # ----------------------------------
        # Update current zone count
        # ----------------------------------

        if current_zone is not None:

            current_zone_counts[
                current_zone
            ] += 1


            # --------------------------------
            # Create history
            # --------------------------------

            if track_id not in person_zone_history:

                person_zone_history[
                    track_id
                ] = set()


            # --------------------------------
            # Count zone visit
            # --------------------------------

            if (
                current_zone
                not in person_zone_history[
                    track_id
                ]
            ):

                zone_visits[
                    current_zone
                ] += 1


                person_zone_history[
                    track_id
                ].add(
                    current_zone
                )


    # --------------------------------------
    # Return analytics
    # --------------------------------------

    return (
        current_zone_counts,
        zone_visits.copy()
    )