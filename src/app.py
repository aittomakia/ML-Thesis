
import sys
import os
# Flask and it's components
import flask
from flask import render_template, send_from_directory, request, redirect,url_for, jsonify
from flask_assets import Environment, Bundle
from src.tf_handler import predict
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
app = app_factory()
app.config['TEMPLATES_AUTO_RELOAD']=True
# app.run(debug=True,use_reloader=True)

UPLOAD_FOLDER = 'src/static/upload'

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['file']
        file_name = secure_filename(upload_file.filename)
        file_name_and_path = os.path.join(UPLOAD_FOLDER, file_name)
        upload_file.save(os.path.join(UPLOAD_FOLDER, file_name))
        # JSON
        # for i in top_k:
        #  output.append({str(labels[i]):str(results[i])})
        output = []
        predictions, results, labels = predict(file_name_and_path, UPLOAD_FOLDER)
        for i in predictions:
            output.append({"label":str(labels[i]), "confidence" :str(round(results[i]*100,2))})

        return render_template('frontpage/index.html', image_path=os.path.join("upload",file_name), output=output)
        #return redirect(url_for('just_upload',pic=filename))
    # return render_template("frontpage")
    # Serve the frontpage
    # return codecs.open("frontend/index.html", 'r').read()


#    app.run(host="0.0.0.0", port=int("5000"), debug=True, use_reloader=False)
