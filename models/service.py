from datetime import datetime
from mongoengine import Document, StringField, DateTimeField, ObjectIdField

class Service(Document):
   # id = ObjectIdField(primary_key=True)
    name = StringField(required=True)
    description = StringField(required=True)
    model_type = StringField(required=True)
    model_name = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)

    def to_dict(self):
        return {
           # 'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'created_at': self.created_at.isoformat()
        }