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
