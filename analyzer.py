import glob
import json
import pprint
from binhex import Error


def pythonify(json_data):
    for key, value in json_data.items():
        if isinstance(value, list):
            value = [pythonify(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            value = pythonify(value)
        try:
            newkey = int(key)
            del json_data[key]
            key = newkey
        except ValueError:
            pass
        json_data[key] = value
    return json_data


def findMiddle(input_list):
    return int(len(input_list) / 2)


json_list = glob.glob("./res/*.json")
for jf in json_list:
    with open(jf) as f_in:
        data_dict = pythonify(json.load(f_in))
        total_test_list = []
        total_train_list = []
        total_valid_list = []
        dict_keys = list(data_dict.keys())
        dict_keys.sort()
        data_needed_keys = dict_keys[0:findMiddle(dict_keys)]
        for k in data_needed_keys:
            total_test_list += data_dict[k]['client_acc'][0]
            total_train_list += data_dict[k]['client_acc'][1]
            total_valid_list += data_dict[k]['client_acc'][2]
        total_test_list.sort()
        total_train_list.sort()
        total_valid_list.sort()
        middle_total_test_list = findMiddle(total_test_list)
        middle_total_train_list = findMiddle(total_train_list)
        middle_total_valid_list = findMiddle(total_valid_list)
        ten_percent_test_list = int(0.1 * (len(total_test_list)))
        ten_percent_train_list = int(0.1 * (len(total_train_list)))
        ten_percent_valid_list = int(0.1 * (len(total_valid_list)))
        worst_avg_test = sum(
            total_test_list[23:23 + ten_percent_test_list]) / \
                         ten_percent_test_list
        worst_avg_train = sum(total_train_list[
                              23:23 + ten_percent_train_list]) / \
                          ten_percent_train_list
        worst_avg_valid = sum(total_valid_list[
                              23:23 + ten_percent_valid_list]) / \
                          ten_percent_valid_list
        best_avg_test = sum(
            total_test_list[len(total_test_list) - ten_percent_test_list:len(total_test_list)]) / \
                        ten_percent_test_list
        best_avg_train = sum(
            total_test_list[len(total_train_list) - ten_percent_train_list:len(total_train_list)]) / \
                         ten_percent_train_list
        best_avg_valid = sum(
            total_test_list[len(total_valid_list) - ten_percent_valid_list:len(total_valid_list)]) / \
                         ten_percent_valid_list

        print(jf)
        print(worst_avg_test)
        print(worst_avg_train)
        print(worst_avg_valid)
        print(best_avg_test)
        print(best_avg_train)
        print(best_avg_valid)
