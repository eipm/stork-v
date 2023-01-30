import re
from functools import partial
from marshmallow import Schema


_snake_case = re.compile(r"(?<=\w)_(\w)")
_to_camel_case = partial(_snake_case.sub, lambda m: m[1].upper())


class CamelCasedSchema(Schema):
    """Gives fields a camelCased data key"""
    def on_bind_field(self, field_name, field_obj, _cc=_to_camel_case):
        field_obj.data_key = _cc(field_name.lower())