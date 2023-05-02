from flask import Blueprint, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from models.user import Role, User
from flask_jwt_extended import get_jwt, get_jti, create_access_token, get_jwt_identity, jwt_required
import redis
from datetime import timedelta
from models.user_rating import UserRating

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

redis_client = redis.Redis(host='localhost', port=6379)



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
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Invalid email or password"}), 401

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]

    redis_client.set(jti, jti, ex=timedelta(days=1))

    return jsonify({'message': 'Logout successful'}), 200


@auth_bp.route("/user", methods=["GET"])
@jwt_required()
def get_current_user():
    user_id = get_jwt_identity()
    user = User.objects(id=user_id).first()
    if not user:
        return {"error": "User not found"}, 404
    return jsonify(user.to_dict()), 200

@auth_bp.route("/users", methods=["GET"])
def get_users():
    users = User.objects.all()
    return jsonify([u.to_dict() for u in users]), 200

@auth_bp.route("/user-delete", methods=["DELETE"])
@jwt_required()
def delete_user():
    user_id = get_jwt_identity()
    user = User.objects(id=user_id).first()
    if not user:
        return {"error": "User not found"}, 404
    user.delete()
    UserRating.objects(user=user).delete()
    return {"message": "User successfully deleted! "}, 200

@auth_bp.route("/user-update", methods=["PUT"])
@jwt_required()
def update_user():
    user_id = get_jwt_identity()
    user = User.objects(id=user_id).first()
    if not user:
        return {"error": "User not found"}, 404
    user.full_name = request.json.get('full_name')
    user.email = request.json.get('email')
    user.password = request.json.get('password')
    user.save()
    return {"message": "User successfully updated!"}, 200
