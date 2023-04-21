from functools import wraps
from flask import session, request, jsonify
from config import Config
from models.user import User
from flask_jwt_extended import JWTManager, get_jwt_identity, jwt_required
import jwt
from flask import current_app

def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        access_token = request.headers.get('Authorization')
        if not access_token:
            return jsonify({'message': 'Access token is missing.'}), 401

        try:
            token = access_token.split()[1]
            data = jwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            user_id = data.get('sub')
            current_user = User.objects(id=user_id).first()
            if token not in session:
                return jsonify({'error': 'Invalid token'}), 401
            if not current_user:
                return jsonify({'message': 'User is not logged in.'}), 401
        except jwt.exceptions.ExpiredSignatureError:
            return jsonify({'message': 'Access token has expired.'}), 401
        except jwt.exceptions.InvalidTokenError:
            return jsonify({'message': 'Invalid access token.'}), 401

        return f(*args, **kwargs)

    return decorated

