import tobii


if __name__ == "__main__":
    print("Tobii接続")
    tobii.connect()

    print("計測開始")
    tobii.start_recording()

    input("Enterキーを押すと計測を停止します。")

    print("計測停止")
    tobii.stop_recording()

    print("終了")