from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignle

class DataController(BaseController):
    
    def __init__(self):
        super().__init__()
        self.size_scale = 1024 * 1024  # Scale size to MB
        
    def validate_uploaded_file(self, file: UploadFile):
        
        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignle.FILE_TYPE_IS_NOT_SUPPORTED.value
        
        if file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, ResponseSignle.FILE_SIZE_EXCEEDED.value
        
        return True, ResponseSignle.FILE_IS_VALID.value
    