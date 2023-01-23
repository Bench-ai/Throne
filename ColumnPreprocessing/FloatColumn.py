import pandas as pd
import numpy as np
import json
from Column import Column
import re
import os
import uuid


class Float:
    method_dict = {
        "check_greater": lambda s, num: Float._gt(s, num),
        "check_lower": lambda s, num: Float._lt(s, num),
        "replace": lambda s, num, rep: Float._replace(s, num, rep),
        "cast": lambda s: s.apply(Float._cast_int),
        "strip": lambda s, rep: s.apply(Float._strip_non_numerics, args=(rep,)),
        "str_to_num_list": lambda s, fp, sep: s.apply(Float._to_json_list, args=(fp, sep))
    }

    valid_instances = (float,
                       np.float64,
                       np.float32,
                       np.float_,
                       np.int32,
                       np.int64,
                       np.int_,
                       int)

    @staticmethod
    def get_methods():
        Column.get_methods(Float.method_dict)

    @staticmethod
    def _gt(s: pd.Series,
            num: float | int) -> pd.Series:

        """
        :param s: The integer Series
        :param num: If a value in the series is less than num it will be turned to NULL
        :return: the fixed series
        """

        if not Column.enforce(Float.valid_instances, s):
            raise TypeError("Data is not consistent, all datatypes must be consistent")

        ser_mask = (s < num)
        s[ser_mask] = np.NaN

        return s

    @staticmethod
    def _lt(s: pd.Series,
            num: int) -> pd.Series:

        """
        :param s: the column Series
        :param num: If a value in the series is greater than num it will be made null
        :return: The fixed series
        """

        if not Column.enforce(Float.valid_instances, s):
            raise TypeError("Data is not consistent, all datatypes must be consistent")

        ser_mask = (s > num)
        s[ser_mask] = np.NaN

        return ser_mask

    @staticmethod
    def _replace(s: pd.Series,
                 num: float | int,
                 rep: float = np.NaN) -> pd.Series:

        if not Column.enforce(Float.valid_instances, s):
            raise TypeError("Data is not consistent, all datatypes must be consistent")

        if not np.isnan(rep) and not isinstance(rep, Float.valid_instances):
            raise TypeError("rep must be a int, float, or NaN")

        """
        :param s: the column series
        :param num: if a value in the series is equal to num it will be replaced with rep
        :param rep: the replacement value
        :return: the fixed series
        """

        ser_mask = (s == num)
        s[ser_mask] = rep

        return ser_mask

    @staticmethod
    def _strip_non_numerics(num: str,
                            replacement: str):

        if isinstance(num, str):
            x = re.sub("[^0-9.]", replacement, num)
            return x
        else:
            return num

    @staticmethod
    def _cast_int(num: str | float):
        try:
            return np.float_(num)
        except ValueError:
            return num

    @staticmethod
    def _to_json_list(line: str,
                      folder_path: str,
                      seperator: str):

        jid = "bench_{}.json".format(uuid.uuid4().hex)
        pt = os.path.join(folder_path,
                          jid)

        num_list = []

        if isinstance(line, str):
            for n in line.split(seperator):
                try:
                    num_list.append(np.float_(n))
                except ValueError:
                    pass
        elif isinstance(line, Float.valid_instances):
            # print(pt)
            num_list.append(line)

        with open(pt, "w") as file:
            json.dump(num_list, file)

        return pt

    @staticmethod
    def process(s: pd.Series,
                pipeline: list[dict]):

        for i in Column.process(pipeline):

            try:
                s = Float.method_dict[i["name"]](s, *list(i["parameters"].values()))
            except TypeError:
                return Column.get_difference(s, Float.valid_instances)[0]

        return s
