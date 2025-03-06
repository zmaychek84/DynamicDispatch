#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

#include <memory.h>
#include <chrono>
#include <random>
#include "qdqlinear_cpu_op.hpp"

#define PROFILING 1
#define TYPE_ int16_t
const int sz_ = 1024 * 10000 + 2;

template <typename T>
void quantize_ref(float* data, T* quantized_data, int size, float scale, int32_t zero_point, int minV, int maxV) {
	for (size_t i = 0; i < size; ++i) {
        int32_t q = std::round(data[i] / scale) + zero_point;
        quantized_data[i] = static_cast<T>(std::clamp(q, minV, maxV)); 
    }
}

template <typename T>
void dequantize_ref(T* quantized_data, float* dequantized_data, int size, float scale, int32_t zero_point) {
    for (size_t i = 0; i < size; ++i) {
        dequantized_data[i] = (static_cast<int32_t>(quantized_data[i]) - zero_point) * scale;
    }
}

float generateRandomFloat(float min, float max) {
    // Scale the result of rand() to the desired range
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

int main() {

    float scale;
    TYPE_ zp;
    bool is_qop;
    bool is_uint16;
    bool is_int8;
    bool is_int16;

    //const float* in_data_f[sz_];
    //uint8_t* out_data_u8[sz_];
    float* in_data_f = (float*)malloc(sz_ * 4);
    TYPE_* out_data = (TYPE_*)malloc(sz_*sizeof(TYPE_));
	float* out_deq = (float*)malloc(sz_ * 4);
	TYPE_* out_data_ref = (TYPE_*)malloc(sz_*sizeof(TYPE_));
	float* out_deq_ref = (float*)malloc(sz_ * 4);

	printf ("Generating input...\n");
	for (int i=0; i<sz_; i++) {
		in_data_f[i] = generateRandomFloat(-55.6,33.4);
	}
	
    scale = 0.01f;
    zp = 127;

    static const int maxval = std::numeric_limits<TYPE_>::max();
    static const int minval = std::numeric_limits<TYPE_>::min();
	
    // -------- Run reference --------
    printf ("Running reference...\n");
	quantize_ref(in_data_f, out_data_ref, sz_, scale, (int32_t)zp, minval, maxval);
	dequantize_ref(out_data_ref, out_deq_ref, sz_, scale, (int32_t)zp);

    printf ("Running CPU op...\n");
    auto THREAD_NUM = std::max((int)std::thread::hardware_concurrency() / 2, (int)1);
    std::vector<std::future<int>> thr_fut(THREAD_NUM);
#if PROFILING
    double t_duration = 0;
    auto t_start = std::chrono::steady_clock::now();
#endif

	cpu_runner_ops::QuantizeLinear(in_data_f, out_data, (std::size_t)sz_, scale, (int)zp, thr_fut);

#if PROFILING					 
    auto t_end = std::chrono::steady_clock::now();
    t_duration += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "Quant latency: " << t_duration << " ms " << std::endl;

    t_duration = 0;
    t_start = std::chrono::steady_clock::now();
#endif

    cpu_runner_ops::DequantizeLinear(out_data, out_deq, (std::size_t)sz_, scale, (int)zp, thr_fut);

#if PROFILING
    t_end = std::chrono::steady_clock::now();
    t_duration += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "Dequant latency: " << t_duration << " ms " << std::endl;
#endif

    // ------- validate ops -------
    printf ("validating ouutput against reference...\n");
    for (int i=0; i<sz_; i++) {
        auto diff = std::abs(out_deq[i] - out_deq_ref[i]);
        if (diff > 0.5f) {
         printf ("diff: %f, out: %f, ref: %f, idx: %d\n",diff,out_deq[i],out_deq_ref[i],i);
         printf ("Test failed!\n");
         return 0;
        }
    }

    printf ("Test passed!\n");
    return -1;
}
