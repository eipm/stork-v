
from marshmallow import fields

from api.models.camel_cased_schema import CamelCasedSchema


class ExperimentResult(CamelCasedSchema):
    # blastocyst_score - predicted BS score from BiLSTM model (ranges from 3 - 14)
    blastocyst_score = fields.Float(data_key="blastocystScore")
    
    # expansion_score - predicted Expansion score from BiLSTM model (ranges from 1 - 4)
    expansion_score = fields.Float(data_key='expansionScore')
    
    # icm_score - predicted ICM score from BiLSTM model (ranges from 1 - 4)
    icm_score = fields.Float(data_key='icmScore')
    
    # trophectoderm_score - predicted Trophectoderm score from BiLSTM model (ranges from 1 - 4)
    trophectoderm_score = fields.Float(data_key='trophectodermScore')
    
    # euploid_probablity - probability for Euploid
    euploid_probablity = fields.Float(data_key='euploidProbablity')
    
    # euploid_prediction - predicted class for Euploid (1 = Euploidy)
    euploid_prediction = fields.Boolean(data_key='euploidPrediction')