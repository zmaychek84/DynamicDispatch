Quantize and Dequantize operator
================================
This test case runs the quant -> dequant on the optimized cpu runner op for size being set in the testcase for random input data,
1. validates against reference OP
2. latency estimate

run command on msvc compiler
============================
powershell:
cl /EHsc /std:c++17 /O2 /arch:AVX512 /Fe:run.exe testcase.cpp; .\run.exe

vs22 developers command prompt:
cl /EHsc /std:c++17 /O2 /arch:AVX512 /Fe:run.exe testcase.cpp && .\run.exe
