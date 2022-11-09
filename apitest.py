from flask import Flask
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
j = {'time': '2018-25-12 23:50:55','name': '王冠博','result': 'ok'}
str1 = json.dumps(j)
@app.route('/', methods=['POST'])
def index():
    return str1

if __name__=='__main__':
    app.run(host='0.0.0.0',port=80, debug=True)