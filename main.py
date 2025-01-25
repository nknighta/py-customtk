import customtkinter as ctk
import cv2
from ultralytics import YOLO
import threading
import time as t
from PIL import Image

# YOLOモデルのロード
yolo = YOLO("models/yolov8n.pt")

# カメラ映像処理クラス
class CameraApp(ctk.CTk):
    def submit_action(self):
            """ボタンが押されたときの動作"""
            input_text = self.entry.get()  # エントリーフィールドから文字列を取得
            if input_text:
                self.label3.configure(text=f"{input_text}様受付完了しました")

    def __init__(self):
        super().__init__()
        self.title("人数カウント")
        self.geometry("700x700")
        
        # 外観設定
        ctk.set_appearance_mode("dark")  # ダークモード
        ctk.set_default_color_theme("blue")  # カラーテーマ
        
        # UI要素の作成
        self.video_label = ctk.CTkLabel(self, text="映像がここに表示されます", width=400, height=300, corner_radius=10)
        self.entry = ctk.CTkEntry(self, placeholder_text="名前を入力(カタカナ)", font=("Helvetica", 20))
        self.label = ctk.CTkLabel(self, text="検出中...", font=("Helvetica", 26))
        self.label2 = ctk.CTkLabel(self, text="入店システム", font=("Helvetica", 16))
        self.label3 = ctk.CTkLabel(self, text="", font=("Helvetica", 26))
        self.button = ctk.CTkButton(self, text="Submit", command=self.submit_action)

        # グリッド配置
        self.video_label.grid(row=2, column=0, columnspan=2, padx=20, pady=20)
        # 入力フォーム
        self.entry.grid(row=3, column=0, padx=20, pady=10)
        self.label.grid(row=3, column=1, padx=20, pady=10)
        self.label2.grid(row=0, column=0, columnspan=2, padx=20, pady=(10, 5))
        # 受けつけ表示
        self.label3.grid(row=1, column=0, columnspan=2, padx=20, pady=(5, 20))
        self.button.grid(row=4, column=0, columnspan=2, padx=20, pady=(5, 20))
        
        # カメラストリーム開始
        self.cap = cv2.VideoCapture(0)
        self.running = True

        # スレッドで映像処理を実行
        self.thread = threading.Thread(target=self.process_video)
        self.thread.start()


    def process_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # YOLOで人検出（クラスID 0は"person"）
            results = yolo(frame)
            detections = results[0].boxes.data.cpu().numpy()  # 検出結果取得

            person_count = sum(1 for det in detections if int(det[-1]) == 0)  # 人数カウント

            # ラベル更新
            self.label.configure(text=f"検出人数: {person_count}(低速動作中)")

            # 映像フレーム描画
            for det in detections:
                if int(det[-1]) == 0:  # 人のみ描画
                    x1, y1, x2, y2 = map(int, det[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # OpenCV画像をPillow形式に変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
            pil_image = Image.fromarray(frame_rgb)  # Pillow形式に変換

            # Pillow画像をCTkImageに変換して表示
            img_tk = ctk.CTkImage(light_image=pil_image, size=(700, 600))  # サイズ調整
            self.video_label.configure(image=img_tk)

            t.sleep(0.1)

    def on_close(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.destroy()

# アプリケーション起動
if __name__ == "__main__":
    app = CameraApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
