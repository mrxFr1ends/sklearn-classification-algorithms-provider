import pickle


def deserialize_object(object):
    return pickle.loads(bytes(object))


def serialize_object(object):
    return list(pickle.dumps(object))


def parse_raw_params(raw_params):
    if raw_params == {}:
        return {}
    params = {}
    for param in raw_params:
        name, value = param.values()
        if value == "true" or value == "false":
            value = value.title()
        try:
            params[name] = eval(value)
        except:
            params[name] = value
        if name == 'estimator' or name == 'base_estimator':
            params[name] = deserialize_object(params[name])
    return params


def get_values_by_shema(data, schema):
    if not schema:
        return data
    if isinstance(data, list):
        return [get_values_by_shema(item, schema) for item in data]

    default_value = None
    has_default = False
    is_required = False
    if isinstance(schema[0], dict):
        key = list(schema[0].keys())[0]
        if 'default' in schema[0][key]:
            has_default = True
            default_value = schema[0][key]['default']
        if 'required' in schema[0][key]:
            is_required = schema[0][key]['required']
            if is_required and not has_default and key not in data:
                raise ValueError(f"No required key '{key}'")
    else:
        key = schema[0]

    if key in data:
        value = data[key]
        if isinstance(value, list):
            return [get_values_by_shema(item, schema[1:]) for item in value]
        else:
            return get_values_by_shema(value, schema[1:])
    else:
        return default_value
