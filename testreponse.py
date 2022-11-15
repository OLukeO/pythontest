from flask import Flask,send_file
app = Flask(__name__)
@app.route("/")
def hello():
    mp4 = send_file("C:/Users/luke/Desktop/Sequence 02.mp4")
    mp4.headers['tok'] = "123456"
    return mp4

if __name__ == "__main__":
    app.run(host="0.0.0.0" ,port=5000)