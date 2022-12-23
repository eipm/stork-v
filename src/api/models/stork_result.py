from marshmallow import fields

from api.models.camel_cased_schema import CamelCasedSchema
from api.models.experiment_result import ExperimentResult


class StorkResult(CamelCasedSchema):
    lr_eup_anu = fields.Nested(ExperimentResult, data_key="lrEupAnu")
    lr_eup_cxa = fields.Nested(ExperimentResult, data_key='lrEupCxa')
