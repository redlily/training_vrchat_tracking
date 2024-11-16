# VRChat用のトラッキングテスト

カメラとVRChatのOSCの仕様を利用したフルボディートラッキング

## VRChat OSC

参考：
 - https://docs.vrchat.com/docs/osc-overview

使用ライブラリ：
 - [python-osc](https://pypi.org/project/python-osc/)
   - VRChatへのトラッキングデータの送信のため

## カメラトラッキング

使用ライブラリ：
 - [opencv-python](https://pypi.org/project/opencv-python/)
   - カメラから動画を取得するため
 - [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=ja)
   - 画像から姿勢を推測するため
 - [NumPy](https://numpy.org/)
   - 計算用

## コントロール

使用ライブラリ：
 - [Flask](https://flask.palletsprojects.com/en/stable/)
   - 外部からプログラムをコントロールするためのインタフェースを作成するため
 