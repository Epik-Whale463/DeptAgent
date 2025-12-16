"""
Data loader for teachers.json
Converts structured teacher data into readable documents for RAG
"""

import json
from typing import List, Dict, Any
from pathlib import Path


class DataLoader:
    """Load and process teacher data from JSON file"""
    
    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load teachers data from JSON file"""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def convert_to_documents(self) -> List[str]:
        """
        Convert structured teacher data into readable text documents
        Each teacher becomes a document with all their information
        """
        teachers = self.load_data()
        documents = []
        
        for teacher in teachers:
            doc = self._format_teacher_document(teacher)
            documents.append(doc)
        
        return documents
    
    @staticmethod
    def _format_teacher_document(teacher: Dict[str, Any]) -> str:
        """Format a single teacher's data into readable document"""
        parts = []
        
        # Structured fields with clear labels
        if "full_name" in teacher:
            parts.append(f"Name: {teacher['full_name']}")
        
        if "department" in teacher:
            parts.append(f"Department: {teacher['department']}")
        
        if "designation" in teacher:
            parts.append(f"Designation: {teacher['designation']}")
        
        if "institution" in teacher:
            parts.append(f"Institution: {teacher['institution']}")
        
        if "employment_type" in teacher:
            parts.append(f"Employment Type: {teacher['employment_type']}")
        
        if "years_of_experience" in teacher:
            parts.append(f"Years of Experience: {teacher['years_of_experience']}")
        
        if "highest_qualification" in teacher:
            parts.append(f"Highest Qualification: {teacher['highest_qualification']}")
        
        if "specialization_area" in teacher:
            parts.append(f"Specialization Area: {teacher['specialization_area']}")
        
        # Add any other fields that might exist
        for key, value in teacher.items():
            if key not in ["professor_id", "full_name", "department", "designation", 
                          "institution", "employment_type", "years_of_experience",
                          "highest_qualification", "specialization_area"]:
                parts.append(f"{key}: {value}")
        
        return "\n".join(parts)
    
    def get_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for each teacher (for retrieval context)"""
        teachers = self.load_data()
        metadata = []
        
        for teacher in teachers:
            meta = {
                "professor_id": teacher.get("professor_id"),
                "full_name": teacher.get("full_name"),
                "department": teacher.get("department"),
                "institution": teacher.get("institution")
            }
            metadata.append(meta)
        
        return metadata
