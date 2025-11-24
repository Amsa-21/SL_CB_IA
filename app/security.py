from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import API_KEY

api_key_header = APIKeyHeader(name="API_KEY")

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante",
        )
