import sys
import json
import csv
import os
import ast

if not len(sys.argv) == 2 or not os.path.isfile(sys.argv[1]):
    print("usage:\n\tpython beautify.py params.out")
    exit()

filename = sys.argv[1]
file = open(filename, 'r')

data = {
    'params': [],
    'scores': []
}

for line in file.readlines():
    if len(line) > 0:
        tmp = line.replace('\n', '').split('\t')
        if len(tmp) == 2:
            params = ast.literal_eval(tmp[0])
            scores = ast.literal_eval(tmp[1].replace('nan', '"nan"'))
            data['params'].append(params)
            data['scores'].append(scores)
file.close()

# print(json.dumps(data, indent=4, separators=(',', ': ')))

# save data to a valid json file
with open("{0}.json".format(filename[:-4]), 'w') as outfile:
    json.dump(data, outfile)

estimators = {}
for i, params_set in  enumerate(data["params"]):
    if params_set["estimator"] not in estimators.keys():
        estimators[params_set["estimator"]] = []
    estimators[params_set["estimator"]].append(i)

def write_to_csv(filename, keys, data, test):
    csv_file = open(filename, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=';')
    csv_writer.writerow(['id'] + keys)
    for i, row in enumerate(data):
        if test(i):
            csv_writer.writerow([i] + [row[key] for key in keys])
    csv_file.close()

write_to_csv('{0}.scores.csv'.format(filename[:-4]),\
             list(data['scores'][0].keys()),\
             data['scores'],\
             lambda x: True)

for estimator in estimators.keys():
    write_to_csv('{0}.params.{1}.csv'.format(filename[:-4], estimator),\
        list(data['params'][estimators[estimator][0]].keys()),\
        data['params'],\
        lambda x: x in estimators[estimator])
