import sys
# setting path
sys.path.append('../stork_v') 

import os
from flask import abort, Flask, request, jsonify, send_from_directory,make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ast import literal_eval
import pathlib
import uuid as myuuid
import json
import datetime

# Relative Imports
from stork_v.stork import *
from api.version import api_version
from api.models.stork_result import *

ALLOWED_EXTENSIONS = set(['zip'])

# Define Directories

current_file_dir = os.path.dirname(os.path.realpath(__file__))

static_file_dir = os.path.join(current_file_dir, '../static')

upload_dir = os.path.join(current_file_dir, '../data')
pathlib.Path(upload_dir).mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

users_dict = literal_eval(os.environ['USERS_DICT'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods = ['GET'])
def serve_index_page():
    return send_from_directory(static_file_dir, 'index.html')

@app.route('/<path:path>', methods = ['GET'])
def serve_assets(path):
    return send_from_directory(static_file_dir, path)

@app.route('/api/healthcheck', methods = ['GET'])
def healthcheck():
    return json.dumps({'status':'Healthy', 'version':api_version()})

@app.route('/login', methods = ['GET'])
def serve_login_page():
    return send_from_directory(static_file_dir, 'login.html')

@app.route('/api/login', methods = ['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username and users_dict[username] == password:
        response = make_response()
        uuid = str(myuuid.uuid4())
        response.set_cookie('stork-auth', uuid, max_age=3600)
        return response
    return json.dumps({}), 401

@app.route('/api/upload', methods = ['POST'])
def predict():
    auth_token = None
    auth_header = request.headers.get('Authorization')

    if auth_header is None or \
        len(auth_header.split(" ")) != 2 or \
            auth_header.split(" ")[1] is None:
        abort(401, 'No Authorization header or authorization failed')

    data = json.loads(request.form['data'])
    
    if data is None:
        abort(400, 'No data part')

    maternal_age =  float(data['maternalAge'])

    # # check if the post request has the file part
    if 'images' not in request.files:
        abort(400, 'No images part')

    if len(request.files.getlist('images')) < 10:
        abort(400, 'A single file is required')
        
    # 1. Create request directory
    request_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    request_dir = (os.path.join(upload_dir, request_id))
    pathlib.Path(request_dir).mkdir(parents=True, exist_ok=True)

    image_paths = []
    # For each uploaded image
    for image in request.files.getlist('images'):
        # 2. Save the zip file
        filename = secure_filename(image.filename)
        image_path = os.path.join(request_dir, filename)
        image_paths.append(image_path)
        image.save(image_path)

    # 3. run
    result = Stork().predict(image_paths, maternal_age, request_id, focus=0)
    
    schema = StorkResult()
    json_response = schema.dump(result)
    return json_response, 200, {'Content-Type': 'application/json; charset=utf-8'}

@app.errorhandler(400)
def bad_request(error):
    body = {
        'error': 'Bad request',
        'message': error.description 
    }
    return make_response(jsonify(body), 400)

@app.errorhandler(401)
def bad_request(error):
    body = {
        'error': 'Unauthorized',
        'message': error.description 
    }
    return make_response(jsonify(body), 401)

