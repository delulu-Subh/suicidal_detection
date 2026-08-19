from pydantic import BaseModel,Field
class PredictionRequest(BaseModel): text:str=Field(min_length=1,max_length=20000)
class FeatureContribution(BaseModel): feature:str;contribution:float
class PredictionResponse(BaseModel): predicted_risk:str;confidence:float;probabilities:dict[str,float];contributing_features:list[FeatureContribution];model_name:str;explanation:str
