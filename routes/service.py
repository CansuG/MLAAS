from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import get_jwt_identity, jwt_required
from mongoengine.errors import ValidationError
from models.service import Service
from flask_security import roles_required, login_required
from decorators import api_login_required
from mongoengine.errors import DoesNotExist

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import pickle
import redis

from models.user import User
from models.user_rating import UserRating


service_bp = Blueprint('service', __name__, url_prefix='/services')

redis_client = redis.Redis(host='localhost', port=6379)

@service_bp.route('/service', methods=['GET'])
def get_services():
    services = Service.objects.all()
    return jsonify([s.to_dict() for s in services]), 200


@service_bp.route('/summarize', methods=['POST'])
def get_summarizer():

    input_text = request.json.get("text")
    model_type = "summarization"
    model_summarizer = pickle.loads(redis_client.get(model_type))
    summary_text = model_summarizer(input_text, max_length=100, min_length=10)[0]["summary_text"]
    try :
        return jsonify({'summary_text' : summary_text}), 200
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400


@service_bp.route('/summarizer', methods=['POST'])
@api_login_required
def create_service():

    #get information of models
    name = request.json.get('name')
    description = request.json.get('description')
    model_name = request.json.get('model_name')
    model_type = request.json.get('model_type')

    #get transformers from hugging face
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer = pipeline(model_type, model=model, tokenizer=tokenizer)
    model_summarizer = pickle.dumps(summarizer)

    #save model redis as pickle
    redis_client.set(model_type, model_summarizer, ex=31536000)

    service = Service(name = name, description = description,model_name= model_name, model_type= model_type )

    try:
        service.save()
        return jsonify(service.to_dict()), 201
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400


@service_bp.route("/rating", methods=["POST"])
@api_login_required
@jwt_required()
def rate_service():
    current_user_id = get_jwt_identity()

    user = User.objects(id=current_user_id).first()
    if not user:
        return {"error": "User not found"}, 404

    service_name = request.json.get("service_name")
    rating = request.json.get("rating")

    service = Service.objects(name=service_name).first()
    if not service:
        return {"error": "Service not found"}, 404

    try:
        user_rating = UserRating.objects.get(user=user, service=service)
        user_rating.rating = rating
    except DoesNotExist:
        user_rating = UserRating(user=user, service=service, rating=rating)

    user_rating.save()

    return {"message": "Rating saved"}, 201


@service_bp.route('/ratings', methods=['GET'])
def get_ratings():
    ratings = UserRating.objects.all()
    return jsonify([s.to_dict() for s in ratings]), 200
