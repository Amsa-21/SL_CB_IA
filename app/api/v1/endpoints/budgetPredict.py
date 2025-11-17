import logging
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def greeting():
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>FastAPI SI-12</title>
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: #22223b;
            }
            h1 {
                text-align: center;
                font-family: Arial, sans-serif;
                color: #f8fafc;
            }
        </style>
    </head>
    <body>
        <h1>FastAPI est en cours d'exécution (Prédiction de budget)...</h1>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
