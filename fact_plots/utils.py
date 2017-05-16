from fact.instrument.camera import camera_distance_mm_to_deg
from fact.analysis.core import (
    split_on_off_source_dependent,
    split_on_off_source_independent,
    default_theta_off_keys
    )

default_theta_off_keys_in_mm = tuple('Theta_Off_{}'.format(i) for i in range(1, 6))


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def generate_theta_deg(df,
                       in_on_key="Theta",
                       in_off_key=default_theta_off_keys_in_mm,
                       out_on_key="Theta_deg",
                       out_off_key=default_theta_off_keys,
                       ):
    '''
    Generate Theta values in deg from theta values
    in mm and add them to the dataframe.
    '''
    df[out_on_key] = df[in_on_key].apply(camera_distance_mm_to_deg)
    for in_key, out_key in zip(in_off_key, out_off_key):
        df[out_key] = df[in_key].apply(camera_distance_mm_to_deg)


def getTheta2Arrays(df, on_key="Theta_deg",
                    prediction_threshold=None,
                    theta_cut=None,
                    is_source_dependend=False
                    ):
    '''
    Get arrays containing the theta values of a given dataframe
    '''
    if is_source_dependend:
        on_data, off_data = split_on_off_source_dependent(
                                df,
                                prediction_threshold=prediction_threshold,
                                )
    else:
        on_data, off_data = split_on_off_source_independent(
                                df,
                                theta_cut=theta_cut
                                )

    theta_on = on_data[on_key].apply(lambda x: x**2)
    theta_off = off_data[on_key].apply(lambda x: x**2)

    return theta_on, theta_off
