import logging
import sys
from blueprints import *
from flask import Flask, request, jsonify

app = Flask(__name__)
app.register_blueprint(homepage)
app.register_blueprint(upload)
 
if __name__ == '__main__':
  app.run(debug=True)

app.debug = True

# App Settings
app.config['UPLOAD_FOLDER'] = 'uploads'
