# Copyright (c) 2024 Advanced Micro Devices, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import os
import shutil
import datetime

def xclbin_transform(test_dir):

    identifier = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    curr_xclbins = []
    new_xclbins = []

    for file_name in os.listdir(test_dir):
        if file_name.endswith(".xclbin"):
            file_path = os.path.join(test_dir, file_name)
            if os.path.isfile(file_path):
                tmp_file_name = file_name[0:-7] + "_tmp_" + identifier + ".xclbin"
                tmp_file_path = os.path.join(test_dir, tmp_file_name)
                os.rename(file_path, tmp_file_path)
                curr_xclbins.append(tmp_file_path)
                new_xclbins.append(file_path)

    print(f"found {len(curr_xclbins)} xclbins")

    for curr_xclbin, new_xclbin in zip(curr_xclbins, new_xclbins):
        print(f"Converting xclbin {new_xclbin}")
        # this rearranges data in PDI so PDI load can be optimized at runtime
        # mainly for legacy xclbins, should be default in newer xclbins
        cmd = f"xclbinutil -i {curr_xclbin} --transform-pdi -o {new_xclbin}"
        os.system(cmd)
        os.remove(curr_xclbin)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help='directory where xclbins are')
    args = parser.parse_args()

    xclbin_transform(args.test_dir)
