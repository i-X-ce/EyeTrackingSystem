import cv2
import json
import time
from screeninfo import get_monitors


with open(
    "config.json",
    "r",
    encoding="utf-8"
) as f:
    config = json.load(f)


def get_image_path(name):
    for image in config["images"]:
        if image["name"] == name:
            return image["file"]

    return None

# 最初に1回だけウィンドウ作成
def init_display():

    # ウィンドウ作成
    cv2.namedWindow(
        "Stimulus",
        cv2.WINDOW_NORMAL
    )

    # 接続されているモニター一覧を取得
    monitors = get_monitors()

    print("接続されているモニター")
    for i, m in enumerate(monitors):
        print(
            f"{i}: "
            f"x={m.x}, y={m.y}, "
            f"width={m.width}, height={m.height}"
        )

    # 2枚目のモニターがあればそこへ移動
    if len(monitors) >= 2:
        second = monitors[1]
        cv2.moveWindow(
            "Stimulus",
            second.x,
            second.y
        )
    else:
        print("外部ディスプレイが見つかりません")

    # フルスクリーン表示
    cv2.setWindowProperty(
        "Stimulus",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )



def show_image(name, seconds):
    path = get_image_path(name)
    img = cv2.imread(path)

    cv2.imshow(
        "Stimulus",
        img
    )

    start = time.time()

    while time.time() - start < seconds:
        cv2.waitKey(1)



def close_display():
    cv2.destroyAllWindows()
