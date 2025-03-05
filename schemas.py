from pydantic import BaseModel
from typing import List, Optional

class Metadata(BaseModel):
    producer: str
    creator: str
    creationdate: str
    source: str
    total_pages: int
    page: int
    page_label: str

class QAPair(BaseModel):
    question: str
    answer: str

class Section(BaseModel):
    section_id: int
    content: str
    metadata: Metadata
    qa_pairs: List[QAPair] 