from flask import Blueprint, jsonify, request, session
from werkzeug.security import generate_password_hash, check_password_hash
from models.user import Role, User
from flask_security import current_user, logout_user
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required
from decorators import api_login_required
from datetime import datetime, timedelta

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    email = request.json.get('email')
    password = request.json.get('password')
    full_name = request.json.get('full_name')
    role = request.json.get('role', Role.USER)

    if email is None or password is None:
        return jsonify({'error': 'email and password are required'}), 400

    if User.objects(email=email).first() is not None:
        return jsonify({'error': 'email already exists'}), 400

    user = User(email=email, password=generate_password_hash(password),full_name=full_name, role=role).save()
    return jsonify({'message': 'user created successfully'}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')

    if not email:
        return jsonify({"msg": "Missing email parameter"}), 400
    if not password:
        return jsonify({"msg": "Missing password parameter"}), 400

    user = User.objects(email=email).first()

    if not user:
        return jsonify({"msg": "User not found"}), 401

    if check_password_hash(user.password, password):
        access_token = create_access_token(identity=str(user.id))
        session[access_token] = access_token
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Invalid email or password"}), 401

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    token = request.headers.get('Authorization').split()[1]
    if token in session:
        session.pop(token, None)
        return jsonify({'message': 'Logout successful'}), 200
    return jsonify({'error': 'Invalid token'}), 401
