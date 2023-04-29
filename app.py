from flask import Flask, jsonify
from flask_mongoengine import MongoEngine
from flask_jwt_extended import JWTManager
from flask_security import Security, MongoEngineUserDatastore
from config import Config
from flask_login import LoginManager
from flask_cors import CORS
import redis 

from models.user import User, Role
from routes.auth import auth_bp
from routes.service import service_bp

db = MongoEngine()
jwt = JWTManager()
security = Security()

app = Flask(__name__)

CORS(app, supports_credentials=True)

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

redis_client = redis.Redis(host='localhost', port=6379)

# The check_if_token_revoked function is used by Flask-JWT-Extended to check whether a token has been revoked.
# In this implementation, it checks if the jti (unique identifier for the token) is in Redis. If it is, then the token has been revoked.
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, decrypted_token):
    jti = decrypted_token['jti']
    token_in_redis = redis_client.get(jti)
    return token_in_redis is not None


# Token revoked callback to handle revoked tokens
@jwt.revoked_token_loader
def revoked_token_callback(jwt_header, jwt_payload):
    return (
        jsonify({"description": "The token has been revoked.", "error": "token_revoked"}),
        401,
    )

# Expired token callback to handle expired tokens
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return (
        jsonify({"message": "The token has expired.", "error": "token_expired"}),
        401,
    )

# Invalid token callback to handle invalid tokens
@jwt.invalid_token_loader
def invalid_token_callback(error):
    return (
        jsonify({"message": "Signature verification failed.", "error": "invalid_token"}),
        401,
    )

# Fresh token required callback to handle non-fresh tokens
@jwt.needs_fresh_token_loader
def token_not_fresh_callback(jwt_header, jwt_payload):
    return (
        jsonify({"description": "The token is not fresh.", "error": "fresh_token_required"}),
        401,
    )

# Unauthorized access callback to handle missing tokens
@jwt.unauthorized_loader
def missing_token_callback(error):
    return (
        jsonify({"description": "Request does not contain an access token.", "error": "authorization_required"}),
        401,
    )


app.register_blueprint(auth_bp)
app.register_blueprint(service_bp)       

if __name__ == '__main__':
    app.run()