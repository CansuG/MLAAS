from werkzeug.security import generate_password_hash, check_password_hash
from mongoengine import Document, CASCADE, ReferenceField, FloatField, StringField, DateTimeField, EmailField, BooleanField, ListField, DoesNotExist
from datetime import datetime
from flask_security import UserMixin, RoleMixin


class Role(Document, RoleMixin):
    name = StringField(max_length=80, unique=True)
    description = StringField(max_length=255)
    

class User(Document, UserMixin):

    email = EmailField(required=True, unique=True)
    password = StringField(required=True)
    full_name = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    roles = ListField(StringField(choices=['admin', 'user']), default=['user'])

    @staticmethod
    def hash_password(password):
        return generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    @classmethod
    def find_by_email(cls, email):
        try:
            return cls.objects.get(email=email)
        except DoesNotExist:
            return None

    def to_dict(self):
        return {
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat()
        }