import tobii_research as tr
import csv
import os
import time


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
    print("計測開始")

    tracker.subscribe_to(
        tr.EYETRACKER_GAZE_DATA,
        gaze_callback,
        as_dictionary=True
    )

def stop_recording():
    print("計測終了")

    tracker.unsubscribe_from(
        tr.EYETRACKER_GAZE_DATA,
        gaze_callback
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(
        "data/gaze_pupil_data.csv",
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

    print("保存完了")
