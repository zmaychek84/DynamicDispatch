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
import os
from pathlib import Path
import zlib
import fnmatch
import multiprocessing as mp
import time


def get_var_name(file_path):
    file_name = file_path.name
    dir_name = file_path.parent.name
    var_name = dir_name + "_" + file_name.split(".")[0]
    return var_name


def bin_to_cpp(file_path):
    variable_name = get_var_name(file_path)
    decompressed_size = 0

    with open(file_path, "rb") as file:
        binary_data = file.read()
        decompressed_size = len(binary_data)
        compressed_data = zlib.compress(binary_data)
        hex_data = "".join(f"\\x{byte:02x}" for byte in compressed_data)
        # Windows has a 16380-char string limit. Break up long strings with ""
        max_string_len = 16380
        hex_data = '""'.join(
            hex_data[i : i + max_string_len]
            for i in range(0, len(hex_data), max_string_len)
        )

    return file_path, variable_name, decompressed_size, len(compressed_data), hex_data


def write_bin_to_file(xclbin_hdrf, xclbin_srcf, data):
    file_path, variable_name, decompressed_size, len_compressed_data, hex_data = data
    with open(xclbin_hdrf, "a") as hdr:
        hdr.write(f"const std::vector<char>& get{variable_name.capitalize()}();\n")

    with open(xclbin_srcf, "a") as src:
        src.write(f'static char {variable_name}[] = "{hex_data}";\n')
        # write a function
        src.write(f"static std::vector<char> initialize_{variable_name}()" + "{\n")
        src.write("std::vector<char> ret;\n")
        src.write("uLongf ret_size = 0;\n")
        src.write(f"ret.resize({decompressed_size});\n")
        src.write("z_stream infstream = {};\n")
        src.write("infstream.zalloc = Z_NULL;\n")
        src.write("infstream.zfree = Z_NULL;\n")
        src.write("infstream.opaque = Z_NULL;\n")
        src.write(f"infstream.avail_in = {len_compressed_data};\n")
        src.write(f"infstream.next_in = reinterpret_cast<Bytef*>({variable_name});\n")
        src.write(f"infstream.avail_out = {decompressed_size};\n")
        src.write(f"infstream.next_out = reinterpret_cast<Bytef*>(&ret[0]);\n")
        src.write("inflateInit(&infstream);\n")
        src.write("inflate(&infstream, Z_NO_FLUSH);\n")
        src.write("inflateEnd(&infstream);\n")
        src.write("return ret;\n")
        src.write("}\n")

        src.write(f"const std::vector<char>& get{variable_name.capitalize()}() {{\n")
        src.write(
            f"static const std::vector<char> {variable_name}_vec = initialize_{variable_name}();\n"
        )
        src.write(f"return {variable_name}_vec;\n")
        src.write("}\n")


def write_xclbin_src(out_dir, xclbin_data, quiet, xclbin_hdr):
    # xclbin_list = [i[0] for i in xclbin_data]
    xclbin_list = xclbin_data
    with open(Path(out_dir) / Path("xclbin_container.cpp"), "w") as xclbin_src:
        xclbin_src.write('#include "xclbin_container.hpp"\n')
        xclbin_src.write(f'#include "{xclbin_hdr}"\n\n')
        xclbin_src.write("XclbinContainer::XclbinContainer(){{}};\n")
        xclbin_src.write(
            "const std::vector<char>& XclbinContainer::get_xclbin_content(const std::string& name) {\n"
        )
        xclbin_src.write("\tstd::string xclbin_name;\n")
        xclbin_src.write("\txclbin_name = name;\n")
        for xclbin_file in xclbin_list:
            var_name = get_var_name(xclbin_file)
            func_name = get_var_name(xclbin_file).capitalize()
            xclbin_src.write(f'\tif (xclbin_name == "{var_name}")\n')
            xclbin_src.write(f"\t\treturn get{func_name}();\n")
        xclbin_src.write("\telse\n")
        xclbin_src.write(
            '\t\tthrow std::runtime_error("Invalid xclbin string name : " + xclbin_name);\n'
        )
        xclbin_src.write("}\n")

        if not quiet:
            print("xclbin_container.cpp is generated")


def get_bin_file():
    for root, dirs, files in os.walk(tr_path):
        for filename in fnmatch.filter(files, "llama2_mladf_2x4x4*.xclbin"):
            file_path = Path(root) / filename
            fname = str(file_path)
            fname = fname.replace("\\", "/")
            yield file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="", type=Path)
    parser.add_argument("--out-dir", required=False, type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root_dir = args.root or Path(os.environ.get("DD_ROOT"))
    out_dir = args.out_dir

    tr_path = root_dir / Path("xclbin/stx")

    xclbin_hdr_fname = "all_xclbin_pkg.hpp"
    xclbin_src_fname = "all_xclbin_pkg.cpp"

    start = time.time()
    if args.list:
        print("xclbin_container.cpp")
        print(xclbin_src_fname)
    else:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        xclbin_list = []
        xclbin_hdrf = Path(out_dir) / Path(xclbin_hdr_fname)
        xclbin_srcf = Path(out_dir) / Path(xclbin_src_fname)
        with open(xclbin_srcf, "w") as src:
            src.write("#include <vector>\n")
            src.write("#include <zlib.h>\n")
            src.write(f'#include "{xclbin_hdr_fname}"\n')

        with open(xclbin_hdrf, "w") as hdr:
            hdr.write("#pragma once\n")
            hdr.write("#include <vector>\n")

        # Process the files in parallel
        # NOTE: pool.imap_unordered can improve performance further, but not tested yet
        with mp.Pool(processes=8) as pool:
            files_iter = get_bin_file()
            xclbin_list_iter = pool.imap(bin_to_cpp, files_iter, chunksize=4)
            xclbin_list = []
            for data in xclbin_list_iter:
                if len(data) == 0:
                    continue
                write_bin_to_file(xclbin_hdrf, xclbin_srcf, data)
                xclbin_list.append(data[0])

        write_xclbin_src(out_dir, xclbin_list, args.quiet, xclbin_hdr_fname)
        if not args.quiet:
            print(f"Header generated: {xclbin_hdr_fname}")
            print(f"Source generated: {xclbin_src_fname}")
    end = time.time()

    if not args.quiet:
        print("Time elapsed (s) : ", end - start)
