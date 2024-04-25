from pydantic import BaseModel
from typing import List
class Model(BaseModel):
    model_name: str = ""
    com_prompt: str = ""
    gen_toxicity: float = 0
    overall_toxicity: float = 0
    com_toxicity: float = 0
    calc_gpt_ppl: float = 0
    diversity: float = 0

class paraDetox(BaseModel):
    com_prompt: str = ""
    paraphrase: str = ""
    com_toxicity: float = 0
    para_toxicity: float = 0
    calc_gpt_ppl: float = 0
    diversity: float = 0
    
class Record(BaseModel):
    pre_prompt: str
    toxicity: float = 0
    models_out: List[Model] = []
    calc_gpt_ppl: float = 0
    paraphrase: paraDetox = None
    diversity: float = 0
