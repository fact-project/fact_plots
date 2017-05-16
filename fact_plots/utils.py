import numpy as np
import pandas as pd
from fact.instrument.camera import camera_distance_mm_to_deg
from fact.analysis.core import (
    split_on_off_source_dependent,
    split_on_off_source_independent,
    default_theta_off_keys
    )

default_theta_off_keys_in_mm = tuple('Theta_Off_{}'.format(i) for i in range(1, 6))


def theta_mm_to_theta_squared_deg(theta):
    return (camera_distance_mm_to_deg(theta))**2


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
