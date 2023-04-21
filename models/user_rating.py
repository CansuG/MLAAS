from mongoengine import Document, CASCADE, ReferenceField, StringField

from .user import User
from .service import Service

class UserRating(Document):
    user = ReferenceField(User, reverse_delete_rule=CASCADE)
    service = ReferenceField(Service, reverse_delete_rule=CASCADE)
    rating = StringField(required=True, choices=['1', '2', '3', '4', '5'])
