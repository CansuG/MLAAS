import json
from flask import Blueprint, jsonify, request, current_app
from flask import send_file
from flask_jwt_extended import current_user, get_jwt_identity, jwt_required
from mongoengine.errors import ValidationError
import requests
from config import Config
from models.service import Service
from flask_security import roles_required, login_required
from mongoengine.errors import DoesNotExist
import os
import cv2
from gender_model.face_recognition import faceRecognitionPipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import base64
from PIL import Image
import numpy as np
import io
import torch

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import pickle
import redis

from models.user import User
from models.user_rating import UserRating


UPLOAD_FOLDER = 'static/upload' 

service_bp = Blueprint('service', __name__, url_prefix='/services')

redis_client = redis.Redis(host='localhost', port=6379)



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

@service_bp.route('/new_service', methods=['POST'])
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

    service = Service(name = name, description = description,model_name= model_name, model_type= model_type )

    try:
        service.save()
        return jsonify(service.to_dict()), 201
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    
@service_bp.route('/gender_classification', methods=['POST'])
def gender_classification():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    f = request.files['file']

    image_pil = Image.open(f)

    # Convert PIL image to OpenCV-compatible format
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR (OpenCV format)


    # get predictions
    pred_image, predictions = faceRecognitionPipeline(image_cv,path=False)

    # Convert the image to Base64-encoded string
    _, img_encoded = cv2.imencode('.jpg', pred_image)

    predicted_image_data = base64.b64encode(img_encoded).decode('utf-8')

    
    return jsonify({'predicted_image_data': predicted_image_data}), 200


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
def create_summarizer():

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

@service_bp.route('/generate_text', methods=['POST'])
def generate_text():

    payload = {
    "inputs": request.json.get('input_text'),
    "max_length": 200,  # Set max_length to generate longer text
    "num_return_sequences": 1
    }

    model_id = "gpt2-large"
    
    api_token = Config.api_token

    headers = {"Authorization": f"Bearer {api_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    try:
        response_data = json.loads(response.content.decode("utf-8"))

        return response_data, 200
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    

# named entity recognition task
@service_bp.route('/ru-en-translation', methods=['POST'])
def ru_2_en_translation():

    input_text = request.json.get('text')

    if not input_text:
        return jsonify({'error': 'Missing input_text field'}), 400   

    payload = {
     "inputs": input_text,
     "parameters": {
            "max_length": 300  # Set the maximum input length here
        } 
    }


    model_id = "Helsinki-NLP/opus-mt-ru-en"
    api_token = Config.api_token

    headers = {"Authorization": f"Bearer {api_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    try:
        response_data = json.loads(response.content.decode("utf-8"))

        return response_data, 200
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400


@service_bp.route('/zero-shot-classification', methods=['POST'])
def zero_shot_classification():

    input_text = request.json.get('text')
    candidate_labels= request.json.get('candidate_labels')

    if not input_text:
        return jsonify({'error': 'Missing input_text field'}), 400   
    

    payload = {
     "inputs": input_text,
     "parameters": {
            "candidate_labels": candidate_labels
        } 
    }


    model_id = "facebook/bart-large-mnli"
    
    api_token = Config.api_token

    headers = {"Authorization": f"Bearer {api_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    try:
        response_data = json.loads(response.content.decode("utf-8"))

        # Get the index of the label with the highest score
        max_score_index = response_data['scores'].index(max(response_data['scores']))

        # Get the corresponding label using the index
        max_score_label = response_data['labels'][max_score_index]

        return jsonify({'label': max_score_label}), 200
    
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    

@service_bp.route('/text-2-text-generation', methods=['POST'])
def text_2_text_generation():

    input_text = request.json.get('text')

    if not input_text:
        return jsonify({'error': 'Missing input_text field'}), 400   
    

    payload = {
     "inputs": input_text
    }


    model_id = "dbmdz/bert-large-cased-finetuned-conll03-english"
    
    api_token = Config.api_token

    headers = {"Authorization": f"Bearer {api_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    try:
        response_data = json.loads(response.content.decode("utf-8"))

        result = []
        for entity in response_data:
            result.append({
                "word": entity["word"],
                "entity_group": entity["entity_group"]
        })

        return jsonify(result), 200
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400