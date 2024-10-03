
# Copyright by OpenCUI, 2024

import json


def parse_json_from_string(text, default=None):
    try:
        return json.loads(text)
    except ValueError as e:
        return default