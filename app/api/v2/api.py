from fastapi import APIRouter

from app.api.v2.endpoints import chatbot, budgetPredict

api_router = APIRouter()
api_router.include_router(chatbot.router, prefix="/chatbot", tags=["Chatbot"])
api_router.include_router(budgetPredict.router, prefix="/prediction", tags=["Prédiction de budget"])