import os
import re
from flask import render_template, request, Blueprint, current_app, send_file,Flask


core = Blueprint("core", __name__)
app = Flask(__name__)
# your request handles here with @core.route()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video", methods=["GET"])
def video():
    headers = request.headers
    if not "range" in headers:
        return current_app.response_class(status=400)

    video_path = os.path.abspath(os.path.join("media", "output.mp4"))
    size = os.stat(video_path)
    size = size.st_size

    chunk_size = 10**3
    start = int(re.sub("\D", "", headers["range"]))
    end = min(start + chunk_size, size - 1)

    content_lenght = end - start + 1

    def get_chunk(video_path, start, end):
        with open(video_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end)
        return chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": content_lenght,
        "Content-Type": "video/mp4",
    }

    return current_app.response_class(get_chunk(video_path, start, end), 206, headers)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)
