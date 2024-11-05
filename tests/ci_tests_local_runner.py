# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

import argparse
import json
import os
import subprocess

def run_cmd(cmd):
    ret = os.system(cmd)
    if ret:
        print(f"Failed to run test: {cmd}")
        os.abort()

def run_test(test):
    run_cmd(test)

def prepare_test_list(test_json):
    test_cmds = []
    with open(test_json) as json_f:
        json_data = json.load(json_f)
        tests = json_data["test_steps"]
        for t in tests:
            test_cmd = t["command"]
            # run only PR tests and not daily regression tests
            if "pr" in t["run_type"]:
                test_cmds.append(test_cmd)

    return test_cmds

def run_all_tests(test_json):
    test_list = prepare_test_list(test_json)
    for test in test_list:
        print(f"Executing : {test}")
        run_test(test)

    return len(test_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", default="", required=True)
    args = parser.parse_args()
    num_tests = run_all_tests(args.test_json)
    print(f"All tests({num_tests} tests) finished successfully! ")
