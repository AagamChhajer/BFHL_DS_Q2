from fastapi import FastAPI
app = FastAPI()

# Import your routes
from .routes import get_lab_tests