from dataclasses import dataclass
from stork_v.dataclasses.experiment_result import *

@dataclass
class StorkResult:
    lr_eup_anu: ExperimentResult
    lr_eup_cxa: ExperimentResult

    def __init__(self, lr_eup_anu: ExperimentResult, lr_eup_cxa: ExperimentResult):
        self.lr_eup_anu = lr_eup_anu
        self.lr_eup_cxa = lr_eup_cxa