# Copyright (c) 2025 Advanced Micro Devices, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    csv_files = [filename for filename in filenames if filename.endswith(".csv")]
    for file in csv_files:
        op_list.clear()
        with open(csv_dir + "/" + file) as csvf:
            csv_reader = csv.reader(csvf)
            for row in csv_reader:
                if len(row) == 0:
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
                    op_list[row[0]] = row[1]

        pdi_op_table[model_identifier] = copy.deepcopy(op_list)
    with open(json_fname, "w", encoding="utf-8") as jsonf:
        jsonf.write(json.dumps(pdi_op_table, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file-dir", default="", required=True)
    parser.add_argument("--json-file", default="", required=True)
    args = parser.parse_args()
    generate_dd_pdi_op_table(args.csv_file_dir, args.json_file)
