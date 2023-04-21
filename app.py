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
login_manager = LoginManager(app)

@jwt.user_lookup_loader
def user_lookup_callback(_jwt_header, jwt_data):
    user_id = jwt_data["sub"]
    return User.objects(id=user_id).first()

jwt = JWTManager(app)
jwt.user_lookup_loader(user_lookup_callback)

user_datastore = MongoEngineUserDatastore(db, User, Role)
security.init_app(app, user_datastore)

app.register_blueprint(auth_bp)
app.register_blueprint(service_bp)       

if __name__ == '__main__':
    app.run()