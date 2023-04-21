from werkzeug.security import generate_password_hash, check_password_hash
from mongoengine import Document, CASCADE, ReferenceField, FloatField, StringField, DateTimeField, EmailField, BooleanField, ListField, DoesNotExist
from datetime import datetime
from flask_security import UserMixin, RoleMixin

from config import Config


class Role:
    ADMIN = 'admin'
    USER = 'user'

class User(Document, UserMixin):

    email = EmailField(required=True, unique=True)
    password = StringField(required=True)
    full_name = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    role = StringField(choices=[Role.ADMIN, Role.USER], default=[Role.USER])

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

    def has_permission(self, permission):
        if self.role == Role.ADMIN:
            return Config.ADMIN_PERMISSIONS.get(permission, False)
        elif self.role == Role.USER:
            return Config.USER_PERMISSIONS.get(permission, False)
        return False

    def to_dict(self):
        return {
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat(),
            'role': self.role
        }