# ==========================================
# RetailAI - Queue Monitoring Engine
# ==========================================

import time


# ==========================================
# QUEUE SETTINGS
# ==========================================

NORMAL_LIMIT = 3
MODERATE_LIMIT = 6
HIGH_LIMIT = 10


# Average estimated time per person
# This is a prototype value and can later
# be calculated from real store data.
AVERAGE_TIME_PER_PERSON = 2


# ==========================================
# QUEUE STATUS
# ==========================================

def get_queue_status(queue_count):

    if queue_count <= NORMAL_LIMIT:

        return "Normal"

    elif queue_count <= MODERATE_LIMIT:

        return "Moderate"

    else:

        return "High"


# ==========================================
# WAITING TIME
# ==========================================

def get_waiting_time(queue_count):

    waiting_time = (
        queue_count *
        AVERAGE_TIME_PER_PERSON
    )

    return waiting_time


# ==========================================
# QUEUE ANALYSIS
# ==========================================

def analyze_queue(queue_count):

    status = get_queue_status(
        queue_count
    )

    waiting_time = get_waiting_time(
        queue_count
    )

    # ======================================
    # ALERT
    # ======================================

    alert = None

    if status == "High":

        alert = (
            "High queue detected. "
            "Consider opening another billing counter."
        )

    elif status == "Moderate":

        alert = (
            "Queue is increasing. "
            "Monitor the billing counter."
        )

    # ======================================
    # RESULT
    # ======================================

    return {

        "people": queue_count,

        "status": status,

        "estimated_wait_minutes":
            waiting_time,

        "alert": alert

    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print("       RetailAI Queue Monitor")
    print("======================================")
    print("")

    test_counts = [
        2,
        5,
        8,
        12
    ]

    for count in test_counts:

        result = analyze_queue(
            count
        )

        print(
            f"Queue: {result['people']} people"
        )

        print(
            f"Status: {result['status']}"
        )

        print(
            f"Estimated Wait: "
            f"{result['estimated_wait_minutes']} minutes"
        )

        if result["alert"]:

            print(
                f"Alert: {result['alert']}"
            )

        else:

            print("Alert: None")

        print("--------------------------------------")