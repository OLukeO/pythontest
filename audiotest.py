import cv2
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, Response
import threading
cap = cv2.VideoCapture(0)


def resize_img_2_bytes(image, resize_factor, quality):
    bytes_io = BytesIO()
    img = Image.fromarray(image)

    w, h = img.size
    img.thumbnail((int(w * resize_factor), int(h * resize_factor)))
    img.save(bytes_io, 'jpeg', quality=quality)

    return bytes_io.getvalue()


def get_image_bytes():
    success, img = cap.read()
    if success:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_bytes = resize_img_2_bytes(img, resize_factor=0.5, quality=30)
        return img_bytes

    return None

app = Flask(
    __name__,
    static_url_path='',
    static_folder='./',
    template_folder='./',
)

@app.route("/", methods=['GET'])
def get_stream_html():
    return render_template('stream.html')

def gen_frames():
    while True:
        img_bytes = get_image_bytes()
        if img_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

def video():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')          # 設定影片的格式為 MJPG
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width,  height))  # 產生空的影片
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        out.write(frame)       # 將取得的每一幀圖像寫入空的影片

    cap.release()
    out.release()      # 釋放資源

@app.route('/api/stream')
def video_stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

while True:
    t = threading.Thread()
    if __name__ == "__main__":
        app.run(host='0.0.0.0')
        video()


