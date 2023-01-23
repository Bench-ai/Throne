from Column import Column
import emoji
import re
import pandas as pd
import string
import json
import uuid
import os


class String:
    valid_instances = (str,)

    method_dict = {
        "strip_non_alnum": lambda s: s.apply(String._only_alpha_numerics),
        "strip_numerics": lambda s: s.apply(String._strip_numerics),
        "strip_punctuation": lambda s: s.apply(String._strip_punc),
        "strip_emoji": lambda s: s.apply(String._strip_emoji),
        "lower": lambda s: s.apply(String._lower),
        "upper": lambda s: s.apply(String._upper),
        "cast": lambda s: s.apply(String._cast),
        "remove_characters": lambda s, char_class: s.apply(String._remove_characters, args=(char_class,)),
        "remove_word": lambda s, char_class: s.apply(String._remove_word, args=(char_class,)),
        "remove_lead_and_trail": lambda s, char_class: s.apply(String._rem_lead_n_end, args=(char_class,)),
        "string_to_json_list": lambda s, folder_path, sep: s.apply(String._rem_lead_n_end, args=(folder_path, sep)),
    }

    @staticmethod
    def get_methods():
        Column.get_methods(String.method_dict)

    @staticmethod
    def _only_alpha_numerics(line: str):

        if isinstance(line, String.valid_instances):
            return re.sub('[\W_]+', '', line)
        else:
            return line

    @staticmethod
    def _strip_numerics(line: str):

        if isinstance(line, String.valid_instances):
            return re.sub("\d+", "", line)
        else:
            return line

    @staticmethod
    def _strip_punc(line):

        if isinstance(line, String.valid_instances):
            return line.translate(str.maketrans('', '', string.punctuation))
        else:
            return line

    @staticmethod
    def _strip_emoji(line):

        if isinstance(line, String.valid_instances):
            return emoji.replace_emoji(line, "")
        else:
            return line

    @staticmethod
    def _lower(line):
        if isinstance(line, String.valid_instances):
            return line.lower()

    @staticmethod
    def _upper(line):
        if isinstance(line, String.valid_instances):
            return line.upper()

    @staticmethod
    def _cast(line):
        try:
            return str(line)
        except ValueError:
            return line

    @staticmethod
    def _remove_characters(line,
                           char_class):

        if isinstance(line, String.valid_instances):
            return re.sub('[{}]'.format(char_class), '', line)

    @staticmethod
    def _remove_word(line,
                     word):

        if isinstance(line, String.valid_instances):
            return line.replace(word, '')

    @staticmethod
    def _rem_lead_n_end(line,
                        char_class):

        if isinstance(line, String.valid_instances):
            return line.strip(char_class)

    @staticmethod
    def _to_json_list(line: str,
                      folder_path: str,
                      seperator: str):

        jid = "bench_{}.json".format(uuid.uuid4().hex)
        pt = os.path.join(folder_path,
                          jid)

        line_list = line.split(seperator)

        with open(pt, "w") as file:
            json.dump(line_list, file)

        return pt

    @staticmethod
    def _to_txt(line: str,
                folder_path: str):

        jid = "bench_{}.txt".format(uuid.uuid4().hex)
        pt = os.path.join(folder_path,
                          jid)

        with open(pt, "w") as file:
            file.write(line)

        return pt

    @staticmethod
    def process(s: pd.Series,
                pipeline: list[dict]):

        for i in Column.process(pipeline):

            try:
                s = String.method_dict[i["name"]](s, *list(i["parameters"].values()))
            except TypeError:
                return Column.get_difference(s, String.valid_instances)[0]

        return s
