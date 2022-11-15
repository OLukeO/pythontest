#import requests

#upload_url = 'https://cs506200.vk.me/upload_video_new.php?act=add_video&mid=21844505&oid=21844505&vid=171170813&fid=0&tag=93bb46ee&hash=e238f469a32fe7eee85a&swfupload=1&api=1'
#file_ = {'file': ('video.mp4', open('video.mp4', 'rb'))}
#r = requests.post(upload_url, files=file_)

#print (r.text)

from flask import Flask
from flask_cors import CORS
import os
import request
import json

app = Flask(__name__)
CORS(app)
file_ = {'video_file': open('C:/Users/luke/PycharmProjects/pythonProject/output.mp4', 'rb')}

@app.route('/', methods=['GET'])
def upload():
    try:
        fileName = '{0}{1}{2}'.format('zz', ''.join(random.sample(char_set, 8)),  '.mp4')
        path = os.path.join('web', app.config['UPLOAD_FOLDER'], fileName)
        url = '/video/{0}'.format(fileName)

        file = request.files['videoFile']

        if file:
            file.save(path)
            return flask.jsonify({'success': True , 'fileName': url})
    except Exception as ex:
        return flask.jsonify({'success': False, 'message': ex})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)