from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import current_user, get_jwt_identity, jwt_required
from mongoengine.errors import ValidationError
from models.service import Service
from flask_security import roles_required, login_required
from mongoengine.errors import DoesNotExist
import os
import cv2
from gender_model.face_recognition import faceRecognitionPipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

UPLOAD_FOLDER = 'static/upload' 



from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import pickle
import redis

from models.user import User
from models.user_rating import UserRating


service_bp = Blueprint('service', __name__, url_prefix='/services')

redis_client = redis.Redis(host='localhost', port=6379)

@service_bp.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    return jsonify({'message': 'Upload successfully'}), 200

@service_bp.route('/services', methods=['GET'])
def get_services():
    services = Service.objects.all()
    return jsonify([s.to_dict() for s in services]), 200


@service_bp.route('/service/<model_name>', methods=['GET'])
def get_service(model_name):
    try:
        service = Service.objects.get(model_name=model_name)
        return jsonify(service.to_dict()), 200
    except DoesNotExist:
        return jsonify({'error': 'Service not found'}), 404



@service_bp.route('/genderapp', methods=['POST'])
def gender_predict():
    file = request.files['file']
    filename = file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    # get predictions
    pred_image, predictions = faceRecognitionPipeline(path)
    # save image
    pred_filename = 'prediction_image.jpg'
    cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)

    return jsonify({'message': 'Prediction was made successfully'}), 200

@service_bp.route('/set-qa', methods=['POST'])
@jwt_required()
def set_question_answering():
    #only admin can add service, permission control
    if not current_user.has_permission('can_create_service'):
        return jsonify({'message': 'Forbidden'}), 403
    
        #get information of models
    name = request.json.get('name')
    description = request.json.get('description')
    model_name = request.json.get('model_name')
    model_type = request.json.get('model_type')

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    question_answerer = pipeline('question-answering', model=model, tokenizer=tokenizer)
    model_question_answerer = pickle.dumps(question_answerer)
   ## result = question_answerer(context=input_text, question=question)

    redis_client.set(model_type, model_question_answerer, ex=31536000)

    qa = Service(name = name, description = description,model_name= model_name, model_type= model_type )

    try:
        qa.save()
        return jsonify(qa.to_dict()), 201
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400



@service_bp.route('/qa', methods=['POST'])
def get_question_answering():

    input_text = request.json.get('text')
    question = request.json.get('question')
  ##  model_name = "distilbert-base-cased-distilled-squad"
    model_type = "question answering"
    question_answerer = pickle.loads(redis_client.get(model_type))
    result = question_answerer(context=input_text, question=question)

    try:
        return jsonify({'answer': result['answer']}), 200
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400

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
@jwt_required()
def create_service():

    #only admin can add service, permission control
    if not current_user.has_permission('can_create_service'):
        return jsonify({'message': 'Forbidden'}), 403
    
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
