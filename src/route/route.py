# -*- coding: utf-8 -*-
"""Public section, including front page."""
from flask import Blueprint, render_template

blueprint = Blueprint('route', __name__)


@blueprint.route('/')
def home():
    """Front page."""
    return render_template('frontpage/index.html')


#@blueprint.route('/about/')
#def about():
#    """About page."""
#return render_template('home/about.html')
