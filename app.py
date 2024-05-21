from flask import Flask, send_from_directory

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)
