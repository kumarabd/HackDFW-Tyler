from flask import Flask, request
from flask_cors import CORS, cross_origin
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/api/input", methods=['GET','POST'])
@cross_origin()
def process_input():
    result = request.get_json()
    return {
        "data": result['data'],
    }