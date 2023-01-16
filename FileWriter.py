import json
import os
from ordered_set import OrderedSet
import copy


class ModelWriter:

    def __call__(self,
                 model_name):

        if not os.path.exists("./ModelDirectory/"):
            raise OSError("Model directory does not exist")

        if not os.path.exists("./ModelDirectory/{}".format(model_name)):
            raise OSError("{} does not exist".format(model_name))

        imp_set = set()
        object_command_list = ["\n",
                               "super({}, self).__init__()\n".format(model_name),
                               "\n"]

        with open("ModelDirectory/{}/{}.json".format(model_name,
                                                     model_name), "r") as j:

            j_dict = json.load(j)

            for dct in j_dict["architecture"]:

                f_n = dct["file_name"]
                imp_str, cls = self.get_layer_import(model_name,
                                                     f_n)
                imp_set.add(imp_str)

                for i in self.get_object_str(cls, dct["parameters"], dct["instance_id"]):
                    object_command_list.append(i)

                object_command_list.append("\n")

        send_list = list(imp_set)
        send_list = ["import torch\n",
                     "import torch.nn as nn\n"] + send_list

        send_list.extend(["\n",
                          "\n",
                          "class {}(nn.Module):\n".format(model_name),
                          "\n",
                          "    def __init__(self):\n"])

        object_command_list = self.shift_str(object_command_list,
                                             8)

        send_list.extend(object_command_list)

        o = self.get_traversal_order(j_dict["architecture"],
                                     j_dict["output_ids"])

        r_list = self.write_forward(o,
                                    j_dict["architecture"],
                                    j_dict["output_ids"])

        send_list.extend(r_list)

        send_list.append("\n")

        self.write_to_file("ModelDirectory/{}/{}.py".format(model_name,
                                                            model_name),
                           send_list)

    @staticmethod
    def get_traversal_order(arch: dict,
                            o_list: list[str]) -> OrderedSet:
        j_data = {}
        o_set = OrderedSet()

        def traverse_json(o_id: list[str],
                          data_dict_list: list[dict]):

            for o, dd in zip(o_id, data_dict_list):

                instance_id, _ = o.split(".")
                dd = copy.deepcopy(dd)

                if instance_id not in o_set:
                    id_list = dd["input_list"]
                    id_list += list(dd["input_dict"].values()) if dd["input_dict"] else []

                    dd_list = [j_data[i_d.split(".")[0]] for i_d in id_list]

                    if len(id_list) == 0:
                        o_set.append(instance_id)
                    else:
                        traverse_json(id_list, dd_list)
                        o_set.append(instance_id)

        arch = copy.deepcopy(arch)

        for i in arch:
            k = i.pop("instance_id")
            j_data[k] = i

        traverse_json(o_list, [j_data[i_d.split(".")[0]] for i_d in o_list])

        return o_set

    @staticmethod
    def get_object_str(class_name: str,
                       parameters: dict,
                       variable_name):

        ret_list = []

        variable_name = variable_name.replace("-", "").lower()

        obj_str = "self._{} = {}("

        obj_str = obj_str.format(variable_name,
                                 class_name)

        ls = list(parameters.items())

        if ls:
            arg_str = "{}={},\n".format(ls[0][0], ls[0][1])

            ret_list.append(obj_str + arg_str)

            for k, v in ls[1:]:
                if isinstance(v, str):
                    v = "'{}'".format(v)

                space_str = " " * len(obj_str)
                ret_list.append(space_str + "{}={},\n".format(k, v))

            ret_list[-1] = ret_list[-1][:-2] + ")\n"
        else:
            obj_str += ")\n"
            ret_list.append(obj_str)

        return ret_list

    @staticmethod
    def get_layer_import(model_name: str,
                         file_name: str) -> tuple[str, str]:

        f_name = file_name[:-3]
        import_str = "from Layers.{} import {}\n"
        layer_name = None

        file_name = os.path.join(".", "ModelDirectory/{}/Layers/{}".format(model_name, f_name + ".py"))

        with open(file_name, "r") as f:
            line = f.readline()

            while line:

                l_split = line.split(" ")

                if l_split[0] == "class":
                    str_len = len(line) * -1
                    layer_name = line[str_len + 6:].split("(")[0]
                    break

                line = f.readline()

        if layer_name is None:
            raise RuntimeError("No Class was found in file {}".format(f_name))

        return import_str.format(f_name, layer_name), layer_name

    @staticmethod
    def write_to_file(file_path: str,
                      command_list: list[str]):
        with open(file_path, "w") as f:
            for c in command_list:
                f.write(c)

    @staticmethod
    def shift_str(command_list: list[str],
                  shift: int) -> list[str]:
        ncl = [(" " * shift) + com for com in command_list]

        return ncl

    @staticmethod
    def write_forward(traversal_path: OrderedSet,
                      json_dict: dict,
                      rets: list[str]):

        data_dict = {}

        for d in json_dict:
            i_d = d.pop("instance_id")
            data_dict[i_d] = d

        str_list = []

        for_str = ((" " * 4) + "def forward(self,\n")
        str_list.append(for_str)

        i_str = (" " * 16 + "in_dict: dict[str, torch.Tensor]):\n")
        str_list.append(i_str)
        str_list.append("\n")

        for layer_id in traversal_path:
            d = data_dict[layer_id]
            entered = False

            if not d["input_dict"] and not d["input_list"]:
                input_dict = "{{'input': in_dict['{}.input']}}".format(layer_id.lower())
                str_input = (" " * 8) + "{} = self._{}(input_dict={},)\n".format("{}_out_dict".format(layer_id.lower()),
                                                                                 layer_id.lower(),
                                                                                 input_dict)

                str_list.append(str_input)

            x = "{} = {}(".format("{}_out_dict".format(layer_id.lower()),
                                  layer_id.lower())

            x_len = len(x) + 1

            if d["input_dict"]:
                entered = True
                it = list(d["input_dict"].items())
                f_line = (" " * 8) + "{} = self._{}(input_dict={{{}: {},\n".format(
                    "{}_out_dict".format(layer_id.lower()),
                    layer_id.lower(),
                    "'" + it[0][0] + "'",
                    it[0][1].replace(".", "_").lower())

                str_list.append(f_line)

                if len(it) > 1:
                    for k, v in it[1:]:
                        cur_str = (" " * (x_len + 8 + 17)) + "{}: {},\n".format("'{}'".format(k),
                                                                                v.replace(".", "_").lower())

                        str_list.append(cur_str)

                str_list[-1] = str_list[-1][:-2] + "},\n"

            if d["input_list"]:

                it = d["input_list"]

                if not entered:
                    f_line = (" " * 8) + "{} = self._{}(input_list=[{},\n".format(
                        "{}_out_dict".format(layer_id.lower()),
                        layer_id.lower(),
                        it[0].replace(".", "_").lower())

                    str_list.append(f_line)
                else:
                    f_line = " " * (x_len + 13) + "input_list=[{},\n".format(it[0].replace(".", "_").lower())
                    str_list.append(f_line)

                if len(it) > 1:

                    for x in it[1:]:
                        f_line = " " * (x_len + 8 + 17) + "{},\n".format(x.replace(".", "_").lower())
                        str_list.append(f_line)

                str_list[-1] = str_list[-1][:-2] + "],\n"

            str_list[-1] = str_list[-1][:-2] + ")\n"

            left_str = ("{}_".format(layer_id.lower()) + "{}, ") * len(d["output"])

            left_str = left_str.format(*d["output"])[:-2]

            right_str = "{}_out_dict".format(layer_id.lower()) + "['{}'], "

            right_str = right_str * len(d["output"])
            right_str = right_str.format(*d["output"])[:-2]

            full_str = "{} = {}\n".format(left_str, right_str)

            str_list.append((" " * 8) + full_str)
            str_list.append("\n")

        r_str = 8 * " " + "return {" + "{}: {},\n".format("'" + rets[0].lower() + "'",
                                                          rets[0].replace(".", "_").lower())

        str_list.append(r_str)
        if len(rets) > 0:
            for i in rets[1:]:
                s = 16 * " " + "{}: {},\n".format("'" + i.lower() + "'",
                                                  i.replace(".", "_").lower())
                str_list.append(s)

        str_list[-1] = str_list[-1][:-2] + "}"

        return str_list
