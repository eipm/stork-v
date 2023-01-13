from dataclasses import dataclass

@dataclass
class ExperimentResult:
    # blastocyst_score - predicted BS score from BiLSTM model (ranges from 3 - 14)
    blastocyst_score: float
    # expansion_score - predicted Expansion score from BiLSTM model (ranges from 1 - 4)
    expansion_score: float
    # icm_score - predicted ICM score from BiLSTM model (ranges from 1 - 4)
    icm_score: float
    # trophectoderm_score - predicted Trophectoderm score from BiLSTM model (ranges from 1 - 4)
    trophectoderm_score: float
    # euploid_probablity - probability for Euploid
    euploid_probablity: float
    # euploid_prediction - predicted class for Euploid (1 = Euploidy)
    euploid_prediction: bool

    def __init__(self, 
                 blastocyst_score: float,
                 expansion_score: float,
                 icm_score: float,
                 trophectoderm_score: float,
                 euploid_probablity: float,
                 euploid_prediction: bool
                 ):
        self.blastocyst_score = blastocyst_score
        self.expansion_score = expansion_score
        self.icm_score = icm_score
        self.trophectoderm_score = trophectoderm_score
        self.euploid_probablity = euploid_probablity
        self.euploid_prediction = euploid_prediction