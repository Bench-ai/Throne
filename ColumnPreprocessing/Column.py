import pandas as pd
import numpy as np


class Column:

    @staticmethod
    def enforce(t: tuple,
                s: pd.Series):

        return Column.get_difference(s, t)[1].shape == s.shape

    @staticmethod
    def get_difference(s: pd.Series,
                       valid_instances: tuple) -> tuple[pd.Series, pd.Series]:

        def mask(val):
            if isinstance(val, valid_instances) or val == np.NaN:
                return True
            else:
                return False

        return s[~s.apply(mask)], s[s.apply(mask)]

    @staticmethod
    def replace_values(index: list[int],
                       replaced_values: list,
                       s: pd.Series,
                       dt):

        full_list = list(zip(index, replaced_values))
        d_type = [('ind', int), ("value", dt)]

        a = np.array(full_list, dtype=d_type)

        full_list.extend(list(s.iteritems()))

        full_list = list(zip(*np.sort(a, order='ind').tolist()))[-1]

        return pd.Series(data=full_list)

    @staticmethod
    def process(pipeline: list[dict]):
        for i in pipeline:
            yield i

    @staticmethod
    def get_methods(method_dict):
        for idx, i in enumerate(method_dict):
            print("{}) {}".format(idx, i))
