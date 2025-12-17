from pydantic import BaseModel, Field

class InitSessionRequest(BaseModel):
    sa_fk: int = Field(..., description="Identifiant unique du SA à analyser.", example=224)
    form_fk: int = Field(..., description="Identifiant unique du formulaire à analyser.", example=167)

class InitSessionResponse(BaseModel):
    session_id: str = Field(..., description="Identifiant unique de la session.", example="a1b2c3d4-e5f6-7890-1234-567890abcdef")

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Identifiant unique de la session.", example="a1b2c3d4-e5f6-7890-1234-567890abcdef")
    question: str = Field(..., description="Question posée par l'utilisateur.", example="Quelle est la formule de l'ebitda ?")
    user_fk: str = Field(..., description="Identifiant de l'utilisateur.", example="8")

class ChatResponse(BaseModel):
    message: str = Field(..., description="Réponse du LLM", example="EBITDA = MARGE 3 + Dotations aux amortissements (CAPEX)")
    response_time: str = Field(..., description="Temps de réponse du LLM", example="2m 35s")

class EndSessionRequest(BaseModel):
    session_id: str = Field(..., description="Identifiant unique de la session à terminer.", example="a1b2c3d4-e5f6-7890-1234-567890abcdef")

class EndSessionResponse(BaseModel):
    message: str = Field(..., description="Message de confirmation de la fin de session.", example="Session terminée avec succès.")

class AnalysisRequest(BaseModel):
    sa_fk: int = Field(..., description="Identifiant unique du SA à analyser.", example=224)

class AnalysisResponse(BaseModel):
    analysis: str = Field(..., description="Analyse textuelle générée.", example="Analyse pour le SA avec l'ID 224. Ceci est un texte d'analyse placeholder.")
    response_time: str = Field(..., description="Temps de réponse du LLM", example="2m 35s")

class WordCloudResponse(BaseModel):
    image: str = Field(..., description="Image encodée en base64 représentant le nuage de mots généré.", example="iVBORw0KGgoAAAANSUhEUgAA...")  # Exemple base64 tronqué