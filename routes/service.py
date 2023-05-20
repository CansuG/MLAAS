from flask import Blueprint, jsonify, request, current_app
from flask import send_file
from flask_jwt_extended import current_user, get_jwt_identity, jwt_required
from mongoengine.errors import ValidationError
import requests
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

@service_bp.route('/set_gender_classification', methods=['POST'])
@jwt_required()
def create_service_gender_classifier():

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

@service_bp.route('/transformers', methods=['POST'])
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

    return send_file(f'./static/predict/{pred_filename}', mimetype='image/jpeg')

   # return jsonify({'message': 'Prediction was made successfully'}), 200

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

@service_bp.route('/set-text-generation', methods=['POST'])
@jwt_required()
def create_text_generator():
    #only admin can add service, permission control
    if not current_user.has_permission('can_create_service'):
        return jsonify({'message': 'Forbidden'}), 403
    
    #get information of models
    name = request.json.get('name')
    description = request.json.get('description')
    model_name = request.json.get('model_name')
    model_type = request.json.get('model_type')

    #gpt2-large
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model=GPT2LMHeadModel.from_pretrained(model_name,pad_token_id=tokenizer.eos_token_id)
    text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    model_text_generator = pickle.dumps(text_generator)

    # Split the model into chunks
    chunk_size = 400 * 1024 * 1024  # 10 MB
    chunks = [model_text_generator[i:i+chunk_size] for i in range(0, len(model_text_generator), chunk_size)]

    # Store the model chunks in Redis
    for i, chunk in enumerate(chunks):
        key = f'model_chunk:{i}'
        redis_client.set(key, chunk)


    redis_client.set(model_type, model_text_generator, ex=31536000)

    text_generator_service = Service(name = name, description = description,model_name= model_name, model_type= model_type )

    try:
        text_generator_service.save()
        return jsonify(text_generator_service.to_dict()), 201
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400


@service_bp.route('/text-generation', methods=['POST'])
def get_text_generator():
    input_text = request.json.get('text')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    model_type = "text_generator"

    model_chunks = []
    num_chunks = len(redis_client.keys("model_chunk:*"))

    for i in range(num_chunks):
        key = f'model_chunk:{i}'
        chunk = redis_client.get(key)
        if chunk is not None:
            model_chunks.append(chunk)

    # Concatenate and reconstruct the model data
    model_data = b''.join(model_chunks)

    # Reconstruct the model from the model data
    try:
        model = torch.load(io.BytesIO(model_data))
    except pickle.UnpicklingError:
        return jsonify({'error': 'Failed to load model'}), 500

    # Generate text using the model
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    try:
        return jsonify({'result': result}), 200
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400



    
# SENTIMENT ANALYSIS

# save sentiment analysis model to redis and database
@service_bp.route('/set-sentiment-analysis', methods=['POST'])
@jwt_required()
def create_sentiment_analysis():
    #only admin can add service, permission control
    if not current_user.has_permission('can_create_service'):
        return jsonify({'message': 'Forbidden'}), 403
    
    #get information of models
    name = request.json.get('name')
    description = request.json.get('description')
    model_name = request.json.get('model_name')
    model_type = request.json.get('model_type')

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    model_sentiment_analysis = pickle.dumps(sentiment_analyzer)
    #result = sentiment_analyzer(input_text)

    redis_client.set(model_type, model_sentiment_analysis, ex=31536000)

    sentiment_analysis = Service(name = name, description = description,model_name= model_name, model_type= model_type )

    try:
        sentiment_analysis.save()
        return jsonify(sentiment_analysis.to_dict()), 201
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400


# use sentiment analysis method
@service_bp.route('/sentiment-analysis', methods=['POST'])
def get_sentiment_analysis():
    
        input_text = request.json.get('text')
        model_type = "sentiment-analysis"
        sentiment_analyzer = pickle.loads(redis_client.get(model_type))
        result = sentiment_analyzer(input_text)
    
        try:
            return jsonify({'sentiment': result[0]['label']}), 200
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