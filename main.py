from display import (
    init_display,
    show_image,
    close_display
)
import tobii


#画像を映す順番を設定
movie_sequence = [
    "01",
    "03",
    "01",
    "02"
]

imageTime = 5
#各画像を何秒流すか設定
display_time = {
    "01":imageTime+imageTime,
    "02":imageTime,
    "03":imageTime,
}

print("display初期化")
init_display()

print("Tobii接続")
tobii.connect()

print("計測開始")
tobii.start_recording()

for name in movie_sequence:
    print("表示:", name)
    show_image(
        name,
        display_time[name]
    )

print("計測停止")
tobii.stop_recording()

print("終了")
close_display()