import asyncio
from functools import lru_cache
import logging

import nest_asyncio
import pandas as pd

from app.services.functions import execute_sp

logger = logging.getLogger(__name__)
nest_asyncio.apply()


def _ensure_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

async def _fetch_questions_dataframe_async(user_fk: int) -> pd.DataFrame:
    lst = await execute_sp(
        "ia.sp_question_get",
        {
            "user_fk": user_fk
        }
    )
    if not lst:
        return pd.DataFrame(columns=["questionLabel", "categorieLabel"])
    return pd.DataFrame(lst)[["questionLabel", "categorieLabel"]]

def _load_questions_dataframe(user_fk: int):
    loop = _ensure_event_loop()
    return loop.run_until_complete(_fetch_questions_dataframe_async(user_fk))

questions = [
    "Quelles sont les principales recettes générées par l'activité ?",
    "Comment évoluent les recettes des logements étudiants ?",
    "Quel est le montant des recettes des appart'hôtels ce mois-ci ?",
    "Quelle part représentent les recettes Activiales dans le total ?",
    "Quelle est la part du CA bar dans le CA global ?",
    "Quel est l'EBITDA enregistré ce mois-ci ?",
    "Comment évolue le Free Cash Flow ?",
    "Quel est le CA net après remises et ristournes ?",
    "Quelle est la part du CA provenant des recettes annexes ?",
    "Quel est le chiffre d'affaires prévisionnel ?",
    "Quel est le CA cumulé sur l'année ?",
    "Quelle est la croissance du CA par rapport à l'année dernière ?",
    "Quel est le CA moyen par résidence ?",
    "Quel est le taux de croissance trimestriel du CA ?",
    "Quelle activité contribue le plus au CA ?",
    "Quel est le chiffre d'affaires total en 2025 ?",
    "Quel était le CA du premier trimestre 2024 ?",
    "Comment a évolué le chiffre d'affaires par rapport à l'année dernière ?",
    "Quel est le CA cumulé au deuxième trimestre 2025 ?",
    "Quel est le total des recettes annexes (boutique, SPA, divers) ?",
    "Quel est le montant des recettes logements étudiants en 2025 ?",
    "Quelles sont les recettes du premier trimestre 2024 ?",
    "Comment ont évolué les recettes commerces par rapport à l'année dernière ?",
    "Quel est le total des recettes Activiales au deuxième trimestre 2025 ?",
    "Quel était le montant des recettes parkings en 2023 ?",
    "Quel est le montant des recettes logements étudiants en janvier 2025 ?",
    "Quel est le total des recettes Activiales en février 2024 ?",
    "Quelle était la recette des commerces au premier trimestre 2023 ?",
    "Quel est le montant des recettes parkings au deuxième trimestre 2025 ?",
    "Quelles recettes ont été générées en mars 2025 ?",
    "Comment les recettes annexes ont-elles évolué en 2024 par rapport à 2023 ?",
    "Quel était le CA bar en avril 2023 ?",
    "Quel est le montant des recettes petits-déjeuners en mai 2025 ?",
    "Quelle était la valeur des recettes SPA au deuxième trimestre 2024 ?",
    "Quel est le total des recettes impayées en juin 2025 ?",
    "Quelle était la part des recettes de commercialisation en 2022 ?",
    "Quel est le montant des recettes annexes au troisième trimestre 2025 ?",
    "Quelles recettes ont augmenté le plus en 2023 par rapport à 2022 ?",
    "Quel est le total des recettes diverses en septembre 2024 ?",
    "Quel était le montant des subventions d'exploitation reçues en 2021 ?",
    
    "Quels sont les postes de charges d'immeuble directes ?",
    "Combien coûte l'électricité des parties communes ?",
    "Quel est le montant des achats alimentaires pour les PDJ ?",
    "Combien représente le poste chauffage ce trimestre ?",
    "Quel est le coût des contrats de maintenance ascenseurs ?",
    "Quels frais sont associés à l'entretien des piscines ?",
    "Quels sont les montants des taxes d'ordures ménagères ?",
    "Quels sont les frais de publicité liés à la gestion Activiales ?",
    "Quel est le total des charges d'assurance ?",
    "Quels frais bancaires ont été enregistrés ?",
    "Quel est le montant des charges liées au personnel ?",
    "Quels honoraires juridiques et contentieux ont été facturés ?",
    "Combien coûtent les fournitures d'accueil des résidences ?",
    "Quels sont les frais de sous-traitance PDJ ?",
    "Quel est le montant des loyers reversés aux propriétaires ?",
    "Combien coûtent les déplacements et missions ?",
    "Quel est le total des frais informatiques ?",
    "Quels impôts et taxes diverses ont été réglés ?",
    "Quels sont les frais financiers sur emprunts PLS ?",
    "Quelle est la part des charges exceptionnelles ?",
    "Quel est le montant des charges d'immeuble directes en 2025 ?",
    "Quelles charges ont le plus augmenté au premier trimestre 2024 ?",
    "Comment ont évolué les charges de personnel depuis l'année dernière ?",
    "Quels étaient les frais bancaires au deuxième trimestre 2025 ?",
    "Quel est le total des charges exceptionnelles en 2023 ?",
    "Quel est le total des charges d'immeuble en janvier 2025 ?",
    "Quelles charges ont augmenté au deuxième trimestre 2024 ?",
    "Quel était le montant des charges de chauffage en hiver 2023 ?",
    "Combien ont coûté les contrats de maintenance en mars 2025 ?",
    "Quel était le total des charges téléphonie en 2022 ?",
    "Quelles charges exceptionnelles ont été enregistrées en avril 2025 ?",
    "Quel est le montant des charges de personnel au premier trimestre 2024 ?",
    "Quelles charges ont diminué en 2023 par rapport à 2022 ?",
    "Quel était le montant des charges financières en mai 2024 ?",
    "Quel est le total des charges de sécurité en juillet 2025 ?",
    "Quel était le coût de l'électricité en août 2023 ?",
    "Combien ont coûté les charges de publicité au troisième trimestre 2024 ?",
    "Quel est le montant des charges fiscales en 2021 ?",
    "Quels étaient les frais bancaires enregistrés en 2023 ?",
    "Quel est le montant des charges sur salaires en octobre 2025 ?",
    
    "Quelle est la valeur de la marge 1 ce mois-ci ?",
    "Comment évolue la marge 2 par rapport au mois précédent ?",
    "Quelle est la marge brute générée par l'activité ?",
    "Quel est le pourcentage de marge par rapport au CA ?",
    "Quelle est la marge nette après déduction des charges ?",
    "Comment se calcule la marge opérationnelle ?",
    "Quel est l'écart entre recettes et charges (marge) ?",
    "Quelle marge est obtenue sur le CA restauration ?",
    "Quelle est la marge réalisée sur les appart'hôtels ?",
    "Quelle est la marge moyenne annuelle ?",
    "Quels sont les facteurs qui influencent la marge ?",
    "Comment se compare la marge 1 et la marge 2 ?",
    "Quel est le taux de marge sur les activités annexes ?",
    "Quelle marge résulte de la gestion des parkings ?",
    "Comment évolue la marge SPA sur 3 mois ?",
    "Quel est l'impact des charges exceptionnelles sur la marge ?",
    "Quel est le solde après calcul de la marge nette ?",
    "Quelle marge prévisionnelle est attendue l'année prochaine ?",
    "Quel est le différentiel entre marge et résultat net ?",
    "Quelle marge est générée par les activités de commercialisation ?",
    "Quelle est la marge 1 réalisée en 2025 ?",
    "Quelle marge nette a été obtenue au premier trimestre 2024 ?",
    "Comment la marge brute a-t-elle évolué par rapport à 2023 ?",
    "Quelle est la marge opérationnelle au deuxième trimestre 2025 ?",
    "Quel était le pourcentage de marge nette l'année dernière ?",
    "Quelle était la marge 1 au premier trimestre 2023 ?",
    "Quelle est la marge nette en janvier 2025 ?",
    "Quel était le pourcentage de marge brute en 2024 ?",
    "Quelle est la marge opérationnelle en février 2025 ?",
    "Quel était le niveau de marge en 2022 par rapport à 2021 ?",
    "Quelle est la marge enregistrée au deuxième trimestre 2024 ?",
    "Quel était le solde de marge en mars 2023 ?",
    "Quelle est la marge SPA en avril 2025 ?",
    "Quelle marge a été obtenue en juin 2024 ?",
    "Quel était le taux de marge nette en septembre 2022 ?",
    "Quelle est la marge du troisième trimestre 2025 ?",
    "Quelle marge brute a été atteinte en décembre 2023 ?",
    "Quel était le différentiel de marge en 2021 ?",
    "Quelle est la marge cumulée sur l'année 2025 ?",
    "Comment la marge a-t-elle évolué entre 2023 et 2024 ?",
    
    "Quelle est la formule du chiffre d'affaires ?",
    "Comment calcule-t-on la marge brute ?",
    "Quelle est la formule du résultat d'exploitation ?",
    "Quelle est la formule du résultat net comptable ?",
    "Comment calcule-t-on la rentabilité économique ?",
    "Quelle est la formule du taux de marge ?",
    "Comment calcule-t-on l'EBITDA ?",
    "Quelle est la formule du Free Cash Flow ?",
    "Quelle est la formule de la capacité d'autofinancement (CAF) ?",
    "Comment calcule-t-on le résultat d'exploitation avant impôt ?",
    "Quelle est la formule de la productivité du capital ?",
    "Quelle est la définition d'un chiffre d'affaires ?",
    "Que signifie une charge d'exploitation ?",
    "Quelle est la définition du CAPEX ?",
    "Comment définir une marge brute ?",
    "Quelle est la définition du free cash flow ?",
    "Que représente le Free Cash Flow ?",
    "Quelle est la définition de l'Ebitda ?",
    "Que signifie le terme amortissement ?",
    "Quelle est la définition d'une charge variable ?",
    "Quelle est la définition d'un coût fixe ?",
    "Quelle est la définition de la rentabilité financière ?",
    "Quelle est la définition de la capacité d'autofinancement ?",
    "Que signifie un excédent brut d'exploitation ?",
    "Quelle est la relation entre chiffre d'affaires et marge brute ?",
    "Comment la marge brute influence-t-elle la marge nette ?",
    "Quelle est la relation entre EBITDA et Free Cash Flow ?",
    "Comment le résultat d'exploitation affecte-t-il le résultat net ?",
    "Quelle est la relation entre charges et résultat ?",
    "Comment les amortissements influencent-ils la rentabilité ?",
    "Comment le chiffre d'affaires impacte-t-il la rentabilité économique ?",
    "Quelle est la relation entre coût de revient et prix de vente ?",
    "Comment les charges fixes et variables influencent-elles le point mort ?",
    "Quelle est la relation entre valeur ajoutée et résultat d'exploitation ?",
    "Quelle est la relation entre dettes et rentabilité financière ?",
    "Comment le niveau d'amortissement impacte-t-il le résultat net ?",
    "Quelle est la relation entre marge d'exploitation et performance globale ?",
    "Comment la croissance du chiffre d'affaires influence-t-elle la marge ?",
    "Quelle est la relation entre charges de personnel et productivité ?",
    "Quelle est la relation entre investissements et cash flow ?",
]

categories = (
    ["Chiffre d'affaire"] * 40
    + ["Charge"] * 40
    + ["Marge"] * 40
    + ["Formule et définition"] * 40
)

@lru_cache(maxsize=4)
def get_questions_and_categories(user_fk: int = 8):
    """Return the static questions list optionally enriched with DB values."""
    questions_list = list(questions)
    categories_list = list(categories)
    try:
        questions_df = _load_questions_dataframe(user_fk)
    except Exception as exc:
        logger.warning("Unable to load questions from DB: %s", exc)
    else:
        if not questions_df.empty:
            questions_list += questions_df["questionLabel"].to_list()
            categories_list += questions_df["categorieLabel"].to_list()

    return questions_list, categories_list