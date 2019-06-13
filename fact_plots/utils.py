def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def kwarg_or_default(kwargs, kwarg_key, default):
    '''
    Use kwarg if kwarg is set else use default value
    :param kwargs: kwargs dict
    :param kwarg_key: key of the kwarg
    :param default: default if kwarg is not set
    :return: value to use
    '''
    return kwargs[kwarg_key] if kwarg_key in kwargs.keys() else default