import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask application
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Load configuration
app.config.from_pyfile('config.py')

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get('DATABASE_URL', app.config.get("SQLALCHEMY_DATABASE_URI"))
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = app.config.get("SQLALCHEMY_TRACK_MODIFICATIONS", False)
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
db.init_app(app)

# Create upload directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['TFLITE_FOLDER'], exist_ok=True)

from routes import *  # Import routes after app is initialized

# Create database tables
with app.app_context():
    db.create_all()
    logger.info("Database tables created")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
