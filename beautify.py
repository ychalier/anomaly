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
            data['scores'].append(params)
file.close()

# print(json.dumps(data, indent=4, separators=(',', ': ')))

# save data to a valid json file
with open("{0}.json".format(filename[:-4]), 'w') as outfile:
    json.dump(data, outfile)
