"""
Disease Information Service for AgriShield Backend (SIH 26131).
Loads and queries actionable agronomic guidance, symptoms, and prevention
for all 38 supported crop disease classes.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging

from .schemas import DiseaseInfo

logger = logging.getLogger("agrishield.api.disease_service")

DISEASE_INFO_PATH = Path(__file__).resolve().parent / "disease_info.json"


class DiseaseService:
    """Manages taxonomic disease information and actionable agricultural guidance."""
    _data: Optional[Dict[str, Dict[str, Any]]] = None

    @classmethod
    def load_data(cls) -> Dict[str, Dict[str, Any]]:
        """Loads disease_info.json once into memory."""
        if cls._data is None:
            if not DISEASE_INFO_PATH.exists():
                raise FileNotFoundError(f"Disease information file not found at: {DISEASE_INFO_PATH}")
            with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
                cls._data = json.load(f)
            logger.info(f"Loaded actionable guidance for {len(cls._data)} crop disease classes.")
        return cls._data

    @classmethod
    def get_all(cls) -> Dict[str, Dict[str, Any]]:
        """Returns full dictionary of all 38 mapped disease classes."""
        return cls.load_data()

    @classmethod
    def find_by_key_or_name(cls, query: str) -> Optional[Dict[str, Any]]:
        """
        Looks up disease information using flexible matching:
        1. Exact match on raw class name (e.g., 'Potato___Early_blight')
        2. Normalized case-insensitive match on raw name
        3. Match on combined '{Crop} {Disease}' (e.g., 'Potato Early Blight')
        4. Match on disease name (e.g., 'Early Blight')
        5. Match on hyphenated slug (e.g., 'potato-early-blight')
        """
        data = cls.load_data()

        # 1. Exact key match
        if query in data:
            return data[query]

        normalized_query = query.strip().lower().replace("-", " ").replace("_", " ")

        # 2. Search through items
        for raw_key, info in data.items():
            raw_normalized = raw_key.lower().replace("-", " ").replace("_", " ")
            combined_name = f"{info.get('crop', '')} {info.get('disease', '')}".strip().lower()
            disease_name = info.get("disease", "").strip().lower()

            if (
                query.lower() == raw_key.lower()
                or normalized_query == raw_normalized
                or normalized_query == combined_name
                or normalized_query == disease_name
            ):
                return info

        return None


def get_disease_info_for_prediction(raw_class_name: str, disease_name: str, crop: str) -> Optional[Dict[str, Any]]:
    """Helper for attaching disease info to prediction response."""
    info = DiseaseService.find_by_key_or_name(raw_class_name)
    if not info:
        info = DiseaseService.find_by_key_or_name(f"{crop} {disease_name}")
    if not info:
        info = DiseaseService.find_by_key_or_name(disease_name)
    return info
