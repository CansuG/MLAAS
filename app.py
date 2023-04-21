from flask import Flask, jsonify
from flask_mongoengine import MongoEngine
from flask_jwt_extended import JWTManager
from flask_security import Security, MongoEngineUserDatastore
from config import Config
from flask_login import LoginManager

from models.user import User, Role
from routes.auth import auth_bp
from routes.service import service_bp

db = MongoEngine()
jwt = JWTManager()
security = Security()

app = Flask(__name__)

app.config.from_object(Config)
db.init_app(app)
jwt.init_app(app)
login_manager = LoginManager(app)

user_datastore = MongoEngineUserDatastore(db, User, Role)
security.init_app(app, user_datastore)

app.register_blueprint(auth_bp)
app.register_blueprint(service_bp)       

if __name__ == '__main__':
    app.run()