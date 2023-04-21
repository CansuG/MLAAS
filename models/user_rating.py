from mongoengine import Document, CASCADE, ReferenceField, StringField, DateTimeField
from datetime import datetime
from .user import User
from .service import Service

class UserRating(Document):
    user = ReferenceField(User, reverse_delete_rule=CASCADE)
    service = ReferenceField(Service, reverse_delete_rule=CASCADE)
    rating = StringField(required=True, choices=['1', '2', '3', '4', '5'])
    rated_at = DateTimeField(default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'user_full_name': str(self.user.full_name),
            'service_name': str(self.service.description),
            'rating': self.rating,
            'rated_at': self.rated_at.isoformat()
        }