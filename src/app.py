
import sys
import os
import json
# Flask and it's components
import flask
from flask import render_template, send_from_directory, request, redirect,url_for, jsonify
from flask_assets import Environment, Bundle
from src.tf_handler import predict, load_graph
from src.image_trace import trace
from flask_cache import Cache
from werkzeug import secure_filename


# -*- coding: utf-8 -*-
"""Create an application instance."""
from src.settings import Config
from src.route import blueprint as route

DEFAULT_BLUEPRINTS = (
    route,
)

def register_blueprints(app, blueprints):
    """Register Flask blueprints."""
    for blueprint in blueprints:
        app.register_blueprint(blueprint)

    return None

def app_factory(blueprints=DEFAULT_BLUEPRINTS):
    app = flask.Flask(__name__)
    # app.config['TEMPLATES_AUTO_RELOAD']=True
    # app.run(debug=True,use_reloader=True)
    register_blueprints(app, blueprints)
    # cache = Cache()
    return app

CONFIG = Config
UPLOAD_FOLDER = 'src/static/upload'
MODEL_FILE = 'src/models/latest/output_graph.pb'

app = app_factory()
app.config['TEMPLATES_AUTO_RELOAD']=True

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['file']
        file_name = secure_filename(upload_file.filename)
        file_name_and_path = os.path.join(UPLOAD_FOLDER, file_name)
        upload_file.save(os.path.join(UPLOAD_FOLDER, file_name))
        # Preload tensorflow graph
        graph = load_graph(MODEL_FILE)

        output = []
        filenames = trace(file_name_and_path, file_name, UPLOAD_FOLDER)
        for file in filenames:
            predictions, results, labels = predict(file_path=UPLOAD_FOLDER + '/' + file, graph=graph)
            for i in predictions:
                # If confidence is higher than 50%
                if round(results[i]*100,2) > 50.00:
                    output.append({"file":str('upload/' + file), "label":str(labels[i]), "confidence" :str(round(results[i]*100,2))})
        return render_template('frontpage/index.html', image_path=os.path.join("upload",file_name), output=output)

@app.route('/images',methods=['POST','GET'])
def images():
    if request.method == 'POST':
        selected_classes = request.form.getlist("images")
        output = []
        for c in selected_classes:
             output.append({"label":str(c)})
        # TODO: remove hardcoded path to the test application
        with open('/Users/sc5-aria/Documents/Projects/DSConf-workshop/demo/components.json', 'w') as outfile:
            json.dump(output, outfile)
        return render_template('frontpage/index.html')
