import tobii_research as tr
import csv
import os
import time
from datetime import datetime


tracker = None
gaze_data_list = []

def connect():
    global tracker
    trackers = tr.find_all_eyetrackers()
    if len(trackers) == 0:
        print("Tobiiなし")
        return False

    tracker = trackers[0]

    print("接続成功")
    print(tracker.model)

    return True

def gaze_callback(gaze_data):

    row = [
        time.time(),

        gaze_data["left_gaze_point_on_display_area"][0],
        gaze_data["left_gaze_point_on_display_area"][1],

        gaze_data["right_gaze_point_on_display_area"][0],
        gaze_data["right_gaze_point_on_display_area"][1],

        gaze_data["left_pupil_diameter"],
        gaze_data["right_pupil_diameter"]
    ]

    gaze_data_list.append(row)

def start_recording():
    tracker.subscribe_to(
        tr.EYETRACKER_GAZE_DATA,
        gaze_callback,
        as_dictionary=True
    )

def stop_recording():
    tracker.unsubscribe_from(
        tr.EYETRACKER_GAZE_DATA,
        gaze_callback
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(
        f"data/gaze_pupil{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
            "Time",
            "LeftGazeX",
            "LeftGazeY",
            "RightGazeX",
            "RightGazeY",
            "LeftPupil",
            "RightPupil"
            ]
        )

        writer.writerows(
            gaze_data_list
        )

