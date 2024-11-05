# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

import argparse
import csv
import copy
import json

import os
import glob

DD_PDI_TABLE_MAJOR_VERSION = 1
DD_PDI_TABLE_MINOR_VERSION = 0
meta_version = str(DD_PDI_TABLE_MAJOR_VERSION) + "." + str(DD_PDI_TABLE_MINOR_VERSION)
def generate_dd_pdi_op_table(csv_dir, json_fname):
    pdi_op_table = {}
    pdi_op_table["DD_PDI_TABLE_MAJOR_VERSION"] = DD_PDI_TABLE_MAJOR_VERSION
    pdi_op_table["DD_PDI_TABLE_MINOR_VERSION"] = DD_PDI_TABLE_MINOR_VERSION
    model_identifier = ""
    op_list = {}

    filenames = os.listdir(csv_dir)
    csv_files = [ filename for filename in filenames if filename.endswith(".csv")]
    for file in csv_files:
        op_list.clear()
        with open(csv_dir + "/" + file) as csvf:
            csv_reader = csv.reader(csvf)
            for row in csv_reader:
                if len(row) ==0:
                    continue

                if row[0] == "DD-PDI-OP-TABLE-VERSION":
                    if row[1] != meta_version:
                        print("Version mismatch. Please check table version")
                        exit()

                elif row[0] == "Model Identifier":
                    model_identifier = row[1]

                elif row[0] == "Op Name":
                    # Ignore op name
                    continue
                else:
                    op_list[row[0]] = (row[1])

        pdi_op_table[model_identifier] = copy.deepcopy(op_list)
    with open(json_fname, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(pdi_op_table, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file-dir", default="", required=True)
    parser.add_argument("--json-file", default="", required=True)
    args = parser.parse_args()
    generate_dd_pdi_op_table(args.csv_file_dir, args.json_file)
