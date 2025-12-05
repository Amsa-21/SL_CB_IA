from fastapi import APIRouter, Depends

from app.api.v1.endpoints import budgetPredict, chatbot
from app.security import get_api_key

api_router = APIRouter()
api_router = APIRouter(dependencies=[Depends(get_api_key)])
api_router.include_router(chatbot.router, prefix="/chatbot", tags=["Chatbot"])
api_router.include_router(budgetPredict.router, prefix="/prediction", tags=["Prédiction de budget"])