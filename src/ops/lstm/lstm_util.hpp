// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <any>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <ops/op_interface.hpp>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

const int MAX_INT32 = ((1LL << 31) - 1);
const int MIN_INT32 = -(1LL << 31);
const int MAX_INT16 = ((1 << 16) - 1);
const int MIN_INT16 = -(1 << 16);
const int FLOAT_DEC_POINT = 24;

class LSTMUtil {

private:
  // Gets specified attribute value if exists
  template <typename KeyT>
  KeyT getAttribute(const std::map<std::string, std::any> &attr,
                    std::string key) {
    KeyT ret_val;

    if ((attr.count(key) == 1) && (attr.at(key).type() == typeid(KeyT))) {
      ret_val = std::any_cast<KeyT>(attr.at(key));
    } else {
      std::cout << key << " not found or not of correct type." << std::endl;
    }

    return ret_val;
  }

  // Define helper functions

  bool is_negative(long long x) { return x < 0; }
  short round_short(double x) {
    bool x_is_negative = is_negative((int64_t)x);
    x = 0.5 + (x_is_negative ? -x : x);
    x = x_is_negative ? -x : x;
    return ((x < MIN_INT16)   ? MIN_INT16
            : (x > MAX_INT16) ? MAX_INT16
                              : static_cast<short>(x));
  }

  // Fills destination buffer with weights after re-arrangement for 8 cores
  void fillWeights(ConstBufferIO &io, int16_t *DstPtr, uint16_t *SrcBasePtr[2],
                   size_t RowLen[2], uint32_t L1TxferSize) {

    size_t CoreOffset = 0;
    size_t SliceOffset = 0;
    size_t GateOffset = 0;

    size_t WRSize = (RowLen[0] + RowLen[1]) * 32;
    size_t DOffset[2] = {0, RowLen[0] * 32}; // For 'W' and 'R'
    size_t DCoreOffset = 0;
    size_t DIdx = 0;

    for (int core = 0; core < 8; core++) { // AIE Cores loop
      GateOffset = CoreOffset;
      DOffset[0] = DCoreOffset;
      DOffset[1] = DCoreOffset + RowLen[0] * 64;

      for (int gate = 0; gate < 4;
           gate++) { // For each gate weights; 4 gates So, 512 / 4 = 128; 128 /
                     // 8 (8 -> # Cores) = 16;

        for (int w_r = 0; w_r < 2;
             w_r++) { // Iterator for W, R; (0-> W, 1 -> R)

          auto SrcPtr = SrcBasePtr[w_r] + (GateOffset * RowLen[w_r]);
          DIdx = DOffset[w_r];
          auto buffer = io.get_buffer((size_t)(DstPtr + DIdx),
                                      2 * RowLen[w_r] * 8 * sizeof(int16_t));
          auto buffer_ptr = (int16_t *)buffer->ptr();
          size_t buffer_idx = 0;
          // Now process 16 rows (done in 2 sets of 8 rows)
          // Process:
          //   1. Each 8 rows need to be read column wise and written as row
          //   major order
          //   2. After the 1st 8 rows repeat the same for next set of 8 rows.
          for (int i = 0; i < 2; i++) {             // For each set of 8-rows
            for (int c = 0; c < RowLen[w_r]; c++) { // Length of each row
              for (int r = 0; r < 8; r++) { // Read 8 rows data column wise
                *(int16_t *)(buffer_ptr + buffer_idx) =
                    (int16_t)((int32_t)SrcPtr[(r * RowLen[w_r]) + c]);
                buffer_idx++;
              }
            }

            SrcPtr += RowLen[w_r] * 8;
          }

          DOffset[w_r] +=
              RowLen[w_r] * 16; // Increment by each gate weight size
        }

        GateOffset += 128;
      }

      CoreOffset += 16;
      DCoreOffset += L1TxferSize;
    }

    return;
  }

  // Finds closest integer (by multiplying float value with 2 powers) so that
  // after
  //      shifting it to right with appropiate no.of bits (2 power value called
  //      shift which is used in closest integer computation) is closer to the
  //      float value.
  template <typename OutT, typename ShiftT, uint8_t MAX_BITS>
  std::pair<OutT, ShiftT> findClosestShiftedIntVal(float Val) {

    const OutT C_MAX_INT = (1 << MAX_BITS) - 1; // Int16 Max is (2^15)-1 = 32767

    float Error, PrevError = 1e9; // A large value
    float IntVal, ModifiedVal = Val;

    ShiftT BestShift = 0, Shift = 0;
    OutT BestIntVal = 0;

    if (Val != 0) { // A condition where no compute is required.
      while (ModifiedVal <=
             C_MAX_INT) { // Loop runs at max 'MAX_BITS + 1' times

        IntVal = std::round(ModifiedVal); // Round returns a float in this case

        Error = std::abs(Val - (IntVal / std::pow(2.0f, (float)Shift))) / Val;

        if (Error < PrevError) {
          PrevError = Error;

          // Save best values
          BestShift = Shift;
          BestIntVal = static_cast<OutT>(IntVal);
        }

        ModifiedVal *= 2;

        Shift++;
      }
    }

    return {BestIntVal, BestShift};
  }

  // Computes DQ, Q paramters for y = x*w + b, by using corresponding scales and
  // zeros points
  void computeMatMulQDQParams(int64_t SumW[], int64_t Bias[], int64_t x_dim,
                              float x_dq_scale, int64_t x_dq_zero_pt,
                              float w_dq_scale, int64_t w_dq_zero_pt,
                              float b_dq_scale, int64_t b_dq_zero_pt,
                              float y_q_scale, int64_t y_q_zero_pt,
                              int64_t C0[], int64_t &C1, int64_t &C2,
                              int64_t &MatMulShift, int64_t &OutShift) {

    // Make x_dim multiple of 32
    x_dim = ((x_dim + 31) / 32) * 32;

    auto [c2_coeff_prime, c2_shift] =
        findClosestShiftedIntVal<int64_t, int16_t, 15>(
            (x_dq_scale * w_dq_scale) / y_q_scale);
    auto [c4_coeff_prime, c4_shift] =
        findClosestShiftedIntVal<int64_t, int16_t, 15>(b_dq_scale / y_q_scale);

    // Adjust c2_coeff_prime so that c2_shift and c4_shift are same
    if (c2_shift > c4_shift) {
      c4_coeff_prime <<= (c2_shift - c4_shift);
    } else {
      c4_coeff_prime >>= (c4_shift - c2_shift);
    }

    int64_t c3_coeff_scale = -c2_coeff_prime * w_dq_zero_pt;
    int32_t c3_coeff_offset = static_cast<int32_t>(-x_dq_zero_pt * x_dim);

    // right shift c3 coeff_scale to ensure fits into int32
    if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
      // Note: Below code is commented because this is an error case so no need
      // to adjust scale uint8_t c3_coeff_scale_shift =
      // static_cast<uint8_t>(std::ceil(std::log2(static_cast<double>(std::abs(c3_coeff_scale))))
      // - 31); c3_coeff_scale = (c3_coeff_scale >> c3_coeff_scale_shift);

      std::cout << "Out-of-range (cannot fit in int32) scale " << c3_coeff_scale
                << std::endl;
      std::cerr << "ERR: Current AIE uint16A_uint16W qdq implementation does "
                   "not support ifm sum shift"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    const int64_t C_MIN = 0, C_MAX = 15;

    MatMulShift = static_cast<int64_t>(std::ceil(std::log2((x_dim))) + 1);
    MatMulShift = std::min(std::max(MatMulShift, C_MIN), C_MAX);

    OutShift = c2_shift;

    C2 = c2_coeff_prime << MatMulShift;
    C1 = c3_coeff_scale;

    // int64_t TmpAdd = (y_q_zero_pt << c2_shift) + (c3_coeff_scale *
    // (c3_coeff_offset << c3_coeff_scale_shift));
    int64_t TmpAdd =
        (y_q_zero_pt << c2_shift) + (c3_coeff_scale * c3_coeff_offset);
    int64_t TmpSumMul = -x_dq_zero_pt * c2_coeff_prime;

    for (int i = 0; i < 512; i++) {
      C0[i] = TmpSumMul * SumW[i] + c4_coeff_prime * Bias[i] + TmpAdd;
    }

    return;
  }

  // Computes parameters and QC values requried for LSTM compuation
  void computeParamsAndQCVals(int16_t *ParamsPtr, int16_t *QC1Ptr,
                              int16_t *QC2Ptr,
                              std::map<std::string, int64_t> &QDQParams,
                              std::map<std::string, float> Scale,
                              std::map<std::string, int64_t> ZeroP,
                              int64_t SumW[], int64_t SumR[], int64_t Bias[],
                              size_t WLen, size_t RLen, int SeqLen) {

    std::map<std::string, int> Params;

    // Scaling factor for floating-point conversion
    Params["s_g"] = 24;

    // For quantized output
    auto qo = 1 / Scale["H"];
    Params["ohs"] = static_cast<int>(std::round(qo));
    Params["ohz"] = static_cast<int>(ZeroP["H"]);
    Params["oh_shift"] = Params["s_g"];
    /*auto qo = 1 / Scale["H"];
    int a = 16 - std::ceil(std::log2(qo));
    std::cout << "q0 = " << qo << "log2(qo) = " << std::log2(qo) << "ceil = " <<
    std::ceil(std::log2(qo)) <<  " a = " << a << std::endl; Params["ohs"] =
    static_cast<int>(std::round(qo * (float)std::pow(2,a))); // TO CHECK *2^a
    Params["ohz"] = static_cast<int>(ZeroP["H"]);
    Params["oh_shift"] = Params["s_g"];   */

    /*Params["oh_shift_old"] = Params["s_g"];
    int shift_old =          Params["oh_shift_old"] ;
    int shift_new =          Params["oh_shift"] ;
    std::cout << " old_s = "   <<  static_cast<int>(std::round(qo *
    (float)std::pow(2,a))) << " new_s = " <<  static_cast<int>(std::round(qo))
    << std::endl; std::cout << " old_shift = "   <<  shift_old << " new_shift =
    " << shift_new << std::endl;*/

    // For quantized output - old
    // auto qo_old = 1 / Scale["H"];
    // int ohs_old = static_cast<int>(std::round(qo_old));
    // int ohz_old = static_cast<int>(ZeroP["H"]);
    // int oh_shift_old = Params["s_g"];
    // std::cout << "\n ohs_old = " << ohs_old << " ohz_old = " << ohz_old << "
    // oh_shift_old = " << oh_shift_old << std::endl;

    // For quantized output
    /*
    double qo = 1 / Scale["H"];
    double scale_tmp = Scale["H"];
    int a = 16 - std::ceil(std::log2(qo));
    //std::cout << " a = " << a << " ceil = " << std::ceil(std::log2(qo))  <<  "
    scale_H = " <<  scale_tmp << " std::log2(qo) = " << std::log2(qo) <<
    std::endl; double s = (a >= 0) ? (1 << a) : (1.0 / ((double)(1 << (-a))));
    s *= qo;
    Params["ohs"] = static_cast<int>(round_short(s));
    Params["ohz"] = static_cast<int>(ZeroP["H"]);
    Params["oh_shift"] = FLOAT_DEC_POINT + a;
    */

    /* int ohz_temp_new = static_cast<int>(ZeroP["H"]);
     std::cout << " \n ohs_new = " << static_cast<int>(round_short(s)) << "
     ohz_new = " << ohz_temp_new << " oh_shift_new = " << FLOAT_DEC_POINT + a <<
     std::endl;
     */

    int ParamsList[] = {0,
                        (int)(WLen),
                        SeqLen,
                        0,
                        0,
                        0,
                        0,
                        Params["s_g"],
                        Params["ohs"],
                        Params["ohz"],
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        Params["oh_shift"]};

    // Packing Params
    for (int i = 0; i < (sizeof(ParamsList) / sizeof(*ParamsList)); i++) {
      if (ParamsList[i] >= ((1 << 15) - 1)) {
        ParamsPtr[i] = (int16_t)(ParamsList[i] - (1 << 16));
      } else {
        ParamsPtr[i] = (int16_t)ParamsList[i];
      }
    }

    // Computing Q-DQ params for MatMul (using conv)
    const int32_t C_2POW24 = 1 << 24;
    int64_t ZeroBias[512] = {0};

    float HScale = 1.0 / C_2POW24;

    int64_t C0_xw[512];
    computeMatMulQDQParams(
        SumW, Bias, WLen, Scale["X"], ZeroP["X"], Scale["W"], ZeroP["W"],
        Scale["B"], ZeroP["B"], HScale, 0, C0_xw, QDQParams["q_a"],
        QDQParams["q_x"], QDQParams["xw_tdm_shift"], QDQParams["xw_out_shift"]);

    int64_t C0_hr[512];
    computeMatMulQDQParams(
        SumR, ZeroBias, RLen, Scale["H"], ZeroP["H"], Scale["R"], ZeroP["R"],
        Scale["H"], 0, HScale, 0, C0_hr, QDQParams["q_b"], QDQParams["q_h"],
        QDQParams["hr_tdm_shift"], QDQParams["hr_out_shift"]);

    // Packing QC1, QC2
    int64_t TmpQC1Val, TmpQC2Val;
    for (int i = 0; i < 8; i++) {      // Core ID
      for (int j = 0; j < 4; j++) {    // Gate ID
        for (int k = 0; k < 16; k++) { // Values per gate per core

          TmpQC1Val = C0_xw[j * 128 + i * 16 + k];
          TmpQC2Val = C0_hr[j * 128 + i * 16 + k];

          for (int l = 0; l < 4;
               l++) { // Each '64-bit' value has 4 '16-bit' values
            QC1Ptr[i * 256 + j * 64 + k * 4 + l] =
                static_cast<int16_t>(TmpQC1Val & 0x000000000000FFFF);
            QC2Ptr[i * 256 + j * 64 + k * 4 + l] =
                static_cast<int16_t>(TmpQC2Val & 0x000000000000FFFF);

            TmpQC1Val >>= 16;
            TmpQC2Val >>= 16;
          }
        }
      }
    }

    return;
  }

  // Packs LSTM Layer paramters into 128 or 64 '16-bit' values
  void packLayerParams(uint8_t *LP_Ptr,
                       std::map<std::string, int64_t> QDQParams,
                       uint16_t NumIter, uint8_t WrapperMode, uint16_t SeqLen,
                       uint16_t W0_Offset, uint16_t R0_Offset,
                       uint16_t W1_Offset, uint16_t R1_Offset,
                       uint16_t ParamsOffset, uint16_t FlipSVWidth,
                       uint16_t WLen, uint16_t RLen, uint8_t Layer,
                       bool FlipMode, uint16_t seqlen_iter) {

    uint8_t KernelWidth = 1;
    uint8_t KernelHeight = 1;
    uint8_t StrideWidth = 1;

    int16_t ifm_depth_gran = 8;
    int16_t ofm_depth_gran = 8;
    int16_t width_gran = 8;
    int16_t height_gran = 1;

    int16_t ifm_bytes = 2;

    int16_t ofm_sv_depth = 64;

    int16_t ifm_sv_height = 1;
    int16_t ofm_sv_width = 8;
    int16_t ofm_sv_height = 1;

    int16_t ofm_size = 1;
    int16_t step_kx = ifm_depth_gran * ifm_bytes;
    int16_t step_Xi = step_kx; // FIXME

    int16_t step_Xo = ofm_depth_gran * ifm_bytes * ofm_size;
    int16_t step_Yo = ofm_sv_width * ofm_sv_depth * ifm_bytes * ofm_size;
    int16_t step_Co = ofm_sv_depth * ofm_depth_gran * ifm_bytes;

    uint8_t zero_init = 1;
    uint8_t sign_N = 0;
    uint8_t sign_O = 0;
    uint8_t sign_W = 0;
    uint8_t sign_A = 0;
    uint8_t skip_casc_in = 0;
    uint8_t skip_casc_out = 0;
    uint8_t norm_ch_g = 0;

    int LoopCnt = FlipMode ? 1 : 2;
    int c1, c2;
    uint16_t inner_g, outer_g;
    if (FlipMode) {
      NumIter = static_cast<uint16_t>(NumIter * seqlen_iter);
      SeqLen = static_cast<uint16_t>(SeqLen / seqlen_iter);
    }

    for (int i = 0; i < LoopCnt; i++) {
      // Loop Specific changes
      int ifm_sv_depth = (i == 0) ? WLen : RLen;

      int16_t ifm_sv_width = (i == 0) ? 8 : 1;
      int16_t step_ky = ifm_sv_width * ifm_sv_depth * ifm_bytes;
      int16_t step_Ci = ifm_sv_width * ifm_depth_gran * ifm_bytes;
      int16_t step_Yi = step_ky; // FIXME

      c1 = static_cast<int>((i == 0) ? QDQParams["q_a"] : QDQParams["q_b"]);
      c2 = static_cast<int>((i == 0) ? QDQParams["q_x"] : QDQParams["q_h"]);

      // Below need to be recomputed as ifm_sv_depth is dependent on LoopCnt
      step_ky = ifm_sv_width * ifm_sv_depth * ifm_bytes;
      step_Yi = step_ky; // FIXME

      if (FlipMode && SeqLen == 24) {
        SeqLen = 20;
      }
      // Filling Layer Parameters
      LP_Ptr[0] = NumIter & 0xFF;
      LP_Ptr[1] = (NumIter >> 8) & 0xFF;
      LP_Ptr[2] = SeqLen & 0xFF;
      LP_Ptr[3] = (SeqLen >> 8) & 0xFF;

      LP_Ptr[4] = KernelWidth;
      LP_Ptr[5] = KernelHeight;
      LP_Ptr[6] = static_cast<uint8_t>(
          std::ceil(ifm_sv_depth / static_cast<double>(ifm_depth_gran)));
      LP_Ptr[7] = StrideWidth;
      LP_Ptr[8] = 1;
      LP_Ptr[9] = static_cast<uint8_t>(
          std::ceil(ofm_sv_width / static_cast<double>(width_gran)));
      LP_Ptr[10] = static_cast<uint8_t>(
          std::ceil(ofm_sv_height / static_cast<double>(height_gran)));
      LP_Ptr[11] = static_cast<uint8_t>(
          std::ceil(ofm_sv_depth / static_cast<double>(ofm_depth_gran)));

      inner_g = LP_Ptr[4] * LP_Ptr[5] * LP_Ptr[6];
      outer_g = LP_Ptr[9] * LP_Ptr[10] * LP_Ptr[11];

      LP_Ptr[12] = inner_g & 0xFF;
      LP_Ptr[13] = (inner_g >> 8) & 0xFF;
      LP_Ptr[14] = outer_g & 0xFF;
      LP_Ptr[15] = (outer_g >> 8) & 0xFF;

      LP_Ptr[16] = static_cast<uint8_t>((i == 0) ? QDQParams["xw_tdm_shift"]
                                                 : QDQParams["hr_tdm_shift"]);
      LP_Ptr[17] = static_cast<uint8_t>((i == 0) ? QDQParams["xw_out_shift"]
                                                 : QDQParams["hr_out_shift"]);
      LP_Ptr[18] = 0; // wts_zp
      LP_Ptr[19] = 9;

      LP_Ptr[20] = step_kx & 0xFF;
      LP_Ptr[21] = (step_kx >> 8) & 0xFF;
      LP_Ptr[22] = step_ky & 0xFF;
      LP_Ptr[23] = (step_ky >> 8) & 0xFF;
      LP_Ptr[24] = step_Ci & 0xFF;
      LP_Ptr[25] = (step_Ci >> 8) & 0xFF;
      LP_Ptr[26] = step_Xi & 0xFF;
      LP_Ptr[27] = (step_Xi >> 8) & 0xFF;
      LP_Ptr[28] = step_Yi & 0xFF;
      LP_Ptr[29] = (step_Yi >> 8) & 0xFF;
      LP_Ptr[30] = step_Xo & 0xFF;
      LP_Ptr[31] = (step_Xo >> 8) & 0xFF;
      LP_Ptr[32] = step_Yo & 0xFF;
      LP_Ptr[33] = (step_Yo >> 8) & 0xFF;
      LP_Ptr[34] = step_Co & 0xFF;
      LP_Ptr[35] = (step_Co >> 8) & 0xFF;

      LP_Ptr[36] = WrapperMode;

      LP_Ptr[40] = (zero_init & 0x01) + ((sign_N << 1) & 0x02) +
                   ((sign_O << 2) & 0x04) + ((skip_casc_in << 6) & 0x40) +
                   ((skip_casc_out << 7) & 0x80);
      LP_Ptr[41] = (sign_W & 0x01) + ((sign_A << 1) & 0x02);
      LP_Ptr[42] = 0;
      LP_Ptr[43] = norm_ch_g;

      LP_Ptr[44] = c1 & 0xFF;
      LP_Ptr[45] = (c1 >> 8) & 0xFF;
      LP_Ptr[46] = (c1 >> 16) & 0xFF;
      LP_Ptr[47] = (c1 >> 24) & 0xFF;
      LP_Ptr[48] = c2 & 0xFF;
      LP_Ptr[49] = (c2 >> 8) & 0xFF;
      LP_Ptr[50] = (c2 >> 16) & 0xFF;
      LP_Ptr[51] = (c2 >> 24) & 0xFF;

      LP_Ptr[54] = (W0_Offset >> 8) & 0xFF;
      LP_Ptr[55] = (R0_Offset >> 8) & 0xFF;
      LP_Ptr[56] = (W1_Offset >> 8) & 0xFF;
      LP_Ptr[57] = (R1_Offset >> 8) & 0xFF;
      LP_Ptr[58] = (ParamsOffset >> 8) & 0xFF;
      LP_Ptr[59] = FlipSVWidth & 0xFF;
      LP_Ptr[60] = (FlipSVWidth >> 8) & 0xFF;
      LP_Ptr[62] = FlipMode ? 0 : Layer;
      LP_Ptr[63] = 0;

      // Prepare for next iteration if required
      LP_Ptr += 64;
    }

    return;
  }

public:
  // LSTMUtil () {
  // }

  //~LSTMUtil () {
  //}

  void generateL1InitData(ConstBufferIO &io, int16_t *dst_ptr,
                          const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr) {

    auto WData = (uint16_t *)const_params.at(0).data;
    auto WShape = const_params.at(0)
                      .shape; // LSTM-0: (2 x 512 x 64), LSTM-1: (2 x 512 x 256)
    auto RData = (uint16_t *)const_params.at(1).data;
    auto RShape = const_params.at(1).shape; // (2 x 512 x 128)
    auto BData = (uint16_t *)const_params.at(2).data;
    auto BShape = const_params.at(2).shape; // (2 x 1024)

    // Processing 'W'/'R' weights
    size_t RowLen[] = {WShape[2], RShape[2]}; // 0: W, 1: R
    uint16_t *SrcBasePtr[2] = {WData, RData};

    bool LSTM0Node = (RowLen[0] != 256); // true: LSTM-0, false: LSTM-1
    std::string lstm_str = LSTM0Node ? "lstm_0" : "lstm_1";
    std::string akeys[] = {(LSTM0Node ? "convfeat" : "lstm_0"),
                           lstm_str + "__W", lstm_str + "__R", lstm_str + "__B",
                           lstm_str};                // Attribute keys
    std::string pkeys[] = {"X", "W", "R", "B", "H"}; // Parameter keys

    // Below sizes are interms of 16-bit values / 2-Bytes / words
    auto WSize = RowLen[0] * 32;
    auto RSize = RowLen[1] * 32;
    auto WRSize = WSize + RSize;
    uint32_t L1TxferSize =
        ((unsigned)WRSize * 2) + 512 + 1024 + 256 +
        256; //(WRSize * 2) + 512 + (2 * ((3 * 128) + (3 * 128))) + 256 + 256;
             // [(W0R0 + W1R1) + (QC:128, H_Init:128, Parms:32, ) + (Non-Linears
             // LUT: tanh-3x128, sigmoid-3x128) + QC1:256 + QC2:256]
    uint32_t LayerParamsSize = (64 * 2) + (32 * 2);

    uint32_t HInitOffset = (unsigned)WRSize * 2;
    uint32_t QCValsOffset = HInitOffset + 128;
    uint32_t ZerosOffset = QCValsOffset + 128;
    uint32_t ParamsOffset = HInitOffset + 448;
    uint32_t LUTOffset = HInitOffset + 512;
    uint32_t QC1ValsOffset =
        LUTOffset + 1024; //(2 * ((3 * 128) + (3 * 128))); // = 3 * 512 = 1536
    uint32_t QC2ValsOffset = QC1ValsOffset + 256;

    /*uint32_t QCValsOffset = WRSize * 2;
    uint32_t HInitOffset  = QCValsOffset + 128;
    uint32_t ZerosOffset  = HInitOffset  + 128;
    uint32_t ParamsOffset = QCValsOffset + 448;
    uint32_t LUTOffset    = QCValsOffset + 512;
    uint32_t QC1ValsOffset = LUTOffset + 1024; //(2 * ((3 * 128) + (3 * 128)));
    // = 3 * 512 = 1536 uint32_t QC2ValsOffset = QC1ValsOffset + 256;*/

    uint32_t LP_lstm_fwd1_Offset = 0; // QC2ValsOffset + 256;
    uint32_t LP_lstm_fwd2_Offset = LP_lstm_fwd1_Offset + 64;
    uint32_t LP_lstm_flip1_Offset = LP_lstm_fwd2_Offset + 64;
    uint32_t LP_lstm_flip2_Offset = LP_lstm_flip1_Offset + 32;

    int16_t SeqLen = getAttribute<int16_t>(attr, "seq_len");

    int16_t Params[64] = {0};
    int16_t HInit[128];
    int16_t Zeros[192] = {0};
    int16_t QC1Vals[4 * 512];
    int16_t QC2Vals[4 * 512];

    int64_t BiasX, BiasH;
    int64_t Bias[512];
    int64_t SumW[512], SumR[512];
    uint32_t RowShift;

    uint16_t FlipSVWidth[2];

    uint16_t seqlen_iter[2];
    seqlen_iter[0] = 1;
    seqlen_iter[1] = 1;

    std::string model_variant =
        getAttribute<std::string>(attr, "model_variant");

    if (SeqLen == 320) { // 1280 Model
      FlipSVWidth[0] = LSTM0Node ? 8 : 16;
      FlipSVWidth[1] = static_cast<uint16_t>(RowLen[1] / 8);
    } else if (SeqLen == 640) { // 2560 Model
      FlipSVWidth[0] = 8;
      FlipSVWidth[1] = 8;
    } else if (SeqLen == 1280) { // 5120 Model
      FlipSVWidth[0] = 8;
      FlipSVWidth[1] = 8;
      seqlen_iter[0] = 4;
      seqlen_iter[1] = 4;
    } else if (SeqLen == 2000) { // 8000 Model
      FlipSVWidth[0] = 8;
      FlipSVWidth[1] = 8;
      seqlen_iter[0] = 8;
      seqlen_iter[1] = 8;
    } else { // For other Models, Using #Cores (= 8)
      FlipSVWidth[0] = static_cast<uint16_t>(RowLen[0] / 8);
      FlipSVWidth[1] = static_cast<uint16_t>(RowLen[1] / 8);
    }
    if (model_variant == "02") // mswbjvw_02
    {
      // mswbjvw_02_2560
      if (SeqLen ==
          320) { // mswbjvw_02_2560  took from mswbjvw -1280 Model as the
                 // lstm1 of pso-02-2560 is same as lstm1 of mswbjvw-1280
        FlipSVWidth[0] =
            LSTM0Node
                ? 16
                : 16; // static_cast<uint16_t>(RowLen[0] / 8);//change here
        FlipSVWidth[1] = static_cast<uint16_t>(RowLen[1] / 8);
      } else if (SeqLen == 640) {
        // mswbjvw_02_5120 -> Model took LSTM1 flipwidths from 2560 model
        FlipSVWidth[0] =
            LSTM0Node
                ? 8
                : 8; // static_cast<uint16_t>(RowLen[0] / 8); //8; change here
        FlipSVWidth[1] =
            LSTM0Node ? 8 : 8; // static_cast<uint16_t>(RowLen[1] / 8); //8;
      } else if (SeqLen == 1000) {
        // 02_8000 Model
        FlipSVWidth[0] = 16;
        FlipSVWidth[1] = 16;
        seqlen_iter[0] = 8;
        seqlen_iter[1] = 8;
        // printf("mswbjvw_02_8000 case.. [%d, %d, %d, %d]\n", FlipSVWidth[0],
        // FlipSVWidth[1], seqlen_iter[0], seqlen_iter[1]);
      } else { // For mswbjvw_02_640
        FlipSVWidth[0] = static_cast<uint16_t>(RowLen[0] / 8);
        FlipSVWidth[1] = static_cast<uint16_t>(RowLen[1] / 8);
      }
    } else if (model_variant == "08") {
      if (SeqLen == 24 || SeqLen == 40) { // 80 Model
        FlipSVWidth[0] =
            static_cast<uint16_t>(RowLen[0] / 8); // LSTM0Node ? 8 : 16;
        FlipSVWidth[1] = static_cast<uint16_t>(RowLen[1] / 8);
      }
    }

    std::map<std::string, float> Scale;
    std::map<std::string, int64_t> ZeroP;
    std::map<std::string, int64_t> QDQParams;

    /*int16_t NonLinearsLUT[1536] = {
        1025,14651,-8116,14675,2318,14704,-2319,14727,1667,14746,31387,14766,-23576,14789,-8812,14815,-29024,14845,-27221,14863,-25730,14882,8914,14904,32539,14928,3682,14956,-25282,14981,15674,14999,10753,15019,-20837,15041,7964,15067,-9834,15095,9100,15116,28254,15134,3672,15155,19368,15178,30227,15204,-3401,15232,32002,15249,3691,15268,-8231,15288,11910,15312,15289,15338,-23002,15363,-11283,15379,-13365,15397,-19889,15417,-21802,15439,-10741,15463,10235,15489,-26941,15503,15553,15519,6261,15536,8009,15554,15131,15573,18290,15593,3718,15614,-23781,15625,18066,15636,-19414,15646,-24374,15656,-19544,15665,-30952,15673,-21856,15679,-22923,15683,161,15685,17357,15683,1907,15678,-1767,15668,-6711,15655,-15668,15638,-19318,15617,9408,15570,-13056,15514,-23275,15421,-32714,15231,-32714,-17537,-23275,-17347,-13056,-17254,9408,-17198,-19318,-17151,-15668,-17130,-6711,-17113,-1767,-17100,1907,-17090,17357,-17085,161,-17083,-22923,-17085,-21856,-17089,-30952,-17095,-19544,-17103,-24374,-17112,-19414,-17122,18066,-17132,-23781,-17143,3718,-17154,18290,-17175,15131,-17195,8009,-17214,6261,-17232,15553,-17249,-26941,-17265,10235,-17279,-10741,-17305,-21802,-17329,-19889,-17351,-13365,-17371,-11283,-17389,-23002,-17405,15289,-17430,11910,-17456,-8231,-17480,3691,-17500,32002,-17519,-3401,-17536,30227,-17564,19368,-17590,3672,-17613,28254,-17634,9100,-17652,-9834,-17673,7964,-17701,-20837,-17727,10753,-17749,15674,-17769,-25282,-17787,3682,-17812,32539,-17840,8914,-17864,-25730,-17886,-27221,-17905,-29024,-17923,-8812,-17953,-23576,-17979,31387,-18002,1667,-18022,-2319,-18041,2318,-18064,-8116,-18093,1025,-18117,-2893,15184,27846,15209,22961,15234,-29633,15249,31304,15266,21676,15285,20833,15306,-21522,15329,-23060,15355,17352,15372,20507,15388,9089,15406,-4675,15425,-7408,15447,15503,15472,-25542,15493,-29389,15508,5604,15525,24058,15543,-28615,15563,-9592,15585,28152,15610,-16550,15626,-24079,15641,-1106,15657,-5460,15675,-29201,15695,1279,15717,28524,15740,-2539,15754,-13216,15768,-14302,15783,-2803,15799,23615,15817,858,15836,-5246,15855,-30696,15874,-19727,15885,27736,15897,-23189,15909,18840,15922,15855,15935,24717,15948,-30662,15961,-31655,15974,7495,15987,5442,15999,5382,16005,2988,16010,19721,16014,-17851,16017,14761,16020,-18100,16021,12730,16022,-22433,16021,12604,16020,-4193,16017,5487,16015,-7965,16011,-26483,16008,-31244,16005,-4302,16002,4178,16001,5446,16000,5446,16000,4178,16001,-4302,16002,-31244,16005,-26483,16008,-7965,16011,5487,16015,-4193,16017,12604,16020,-22433,16021,12730,16022,-18100,16021,14761,16020,-17851,16017,19721,16014,2988,16010,5382,16005,5442,15999,7495,15987,-31655,15974,-30662,15961,24717,15948,15855,15935,18840,15922,-23189,15909,27736,15897,-19727,15885,-30696,15874,-5246,15855,858,15836,23615,15817,-2803,15799,-14302,15783,-13216,15768,-2539,15754,28524,15740,1279,15717,-29201,15695,-5460,15675,-1106,15657,-24079,15641,-16550,15626,28152,15610,-9592,15585,-28615,15563,24058,15543,5604,15525,-29389,15508,-25542,15493,15503,15472,-7408,15447,-4675,15425,9089,15406,20507,15388,17352,15372,-23060,15355,-21522,15329,20833,15306,21676,15285,31304,15266,-29633,15249,22961,15234,27846,15209,-2893,15184,
        25588,15468,15538,15490,29305,15503,-4561,15517,-13103,15533,11663,15551,12780,15570,-552,15590,-18488,15613,17510,15627,-12723,15640,-28464,15655,-23420,15671,9045,15689,10369,15708,-12152,15728,-25462,15747,-16278,15759,-7252,15772,5731,15787,26829,15802,-5345,15818,-21172,15836,-16711,15855,5879,15874,1044,15885,-21391,15896,5288,15909,16347,15922,12125,15936,-7607,15950,21785,15966,-32472,15982,23609,15999,27641,16008,28847,16017,-20419,16026,7141,16036,-24053,16045,11816,16055,-22313,16064,-2035,16073,-128,16082,-24254,16091,-16731,16099,14686,16107,-2841,16113,-10207,16119,-12515,16124,26224,16128,-6516,16129,-5455,16130,31489,16131,-23323,16131,31357,16131,4479,16131,-32362,16130,-7602,16129,18644,16129,-14894,16128,25525,16128,9651,16128,2004,16128,0,16128,0,16128,-4009,16127,-19302,16127,14486,16127,29787,16126,28248,16125,15205,16124,-812,16122,-8959,16121,2823,16121,-18890,16120,2559,16121,10910,16122,13031,16124,13087,16127,-26510,16129,5103,16132,1421,16135,25425,16138,8365,16142,12127,16146,-32704,16150,1017,16155,-21611,16159,26860,16164,12026,16169,-3570,16173,-22559,16178,18344,16183,-13821,16187,10482,16192,24502,16196,27322,16200,18286,16204,-3031,16207,28681,16211,-17706,16214,-11036,16217,-16645,16220,31298,16223,2089,16226,27223,16228,-23908,16230,-19738,16232,-25292,16234,25482,16236,2035,16238,-29585,16239,-3336,16240,15736,16242,28107,16243,-31304,16244,-30989,16245,29467,16246,19386,16247,4674,16248,-14319,16248,28273,16249,1684,16250,-28263,16250,4239,16251,-31636,16251,-4582,16251,20080,16252,6610,13577,2684,13616,2688,13666,7941,13713,22231,13754,17314,13807,-25521,13849,15713,13893,17082,13949,-26452,13986,-14576,14032,2466,14086,7034,14124,-685,14172,-8024,14221,11372,14262,-5637,14313,11440,14358,-11404,14400,-26855,14455,-3054,14494,6278,14540,1865,14595,15590,14632,777,14680,-21350,14730,3133,14770,-26490,14820,-16831,14866,25685,14908,-9798,14961,15091,15003,16537,15047,-17420,15103,6111,15140,-29530,15186,3371,15239,13458,15277,4197,15326,19102,15374,16461,15414,17556,15465,9664,15509,-32572,15550,-2423,15602,-23670,15642,23901,15684,-28266,15736,-19947,15772,-29598,15812,-4779,15860,22083,15895,2487,15929,13236,15967,21259,16004,-32021,16025,15070,16045,-18550,16060,-32406,16068,-8068,16064,-20325,16046,30784,16012,-12400,15926,865,15742,865,-17026,-12400,-16842,30784,-16756,-20325,-16722,-8068,-16704,-32406,-16700,-18550,-16708,15070,-16723,-32021,-16743,21259,-16764,13236,-16801,2487,-16839,22083,-16873,-4779,-16908,-29598,-16956,-19947,-16996,-28266,-17032,23901,-17084,-23670,-17126,-2423,-17166,-32572,-17218,9664,-17259,17556,-17303,16461,-17354,19102,-17394,4197,-17442,13458,-17491,3371,-17529,-29530,-17582,6111,-17628,-17420,-17665,16537,-17721,15091,-17765,-9798,-17807,25685,-17860,-16831,-17902,-26490,-17948,3133,-17998,-21350,-18038,777,-18088,15590,-18136,1865,-18173,6278,-18228,-3054,-18274,-26855,-18313,-11404,-18368,11440,-18410,-5637,-18455,11372,-18506,-8024,-18547,-685,-18596,7034,-18644,2466,-18682,-14576,-18736,-26452,-18782,17082,-18819,15713,-18875,-25521,-18919,17314,-18961,22231,-19014,7941,-19055,2688,-19102,2684,-19152,6610,-19191,
        -25578,14096,-4464,14134,23413,14183,17481,14226,-6645,14264,-21239,14313,-24729,14355,30754,14394,31057,14443,-23822,14484,-26827,14523,-20836,14572,17261,14613,13401,14652,14732,14701,29764,14741,16181,14780,1245,14829,10146,14869,-23097,14907,-1765,14955,20026,14996,21525,15034,-991,15081,-10517,15122,13209,15160,-4030,15206,-20972,15248,10822,15285,-19781,15330,-17306,15373,6311,15409,6354,15453,-7003,15497,-10644,15531,-5509,15573,-118,15620,11281,15653,-10074,15692,-22338,15741,-14401,15772,27805,15809,6534,15854,11148,15890,-4361,15922,18199,15962,-26795,16004,19794,16032,-20548,16064,1123,16102,10266,16136,-26382,16159,-9531,16184,5012,16211,-1867,16236,26530,16258,17997,16268,4789,16275,2044,16278,-3357,16276,30369,16272,7187,16266,1867,16260,21608,16256,21608,16256,1867,16260,7187,16266,30369,16272,-3357,16276,2044,16278,4789,16275,17997,16268,26530,16258,-1867,16236,5012,16211,-9531,16184,-26382,16159,10266,16136,1123,16102,-20548,16064,19794,16032,-26795,16004,18199,15962,-4361,15922,11148,15890,6534,15854,27805,15809,-14401,15772,-22338,15741,-10074,15692,11281,15653,-118,15620,-5509,15573,-10644,15531,-7003,15497,6354,15453,6311,15409,-17306,15373,-19781,15330,10822,15285,-20972,15248,-4030,15206,13209,15160,-10517,15122,-991,15081,21525,15034,20026,14996,-1765,14955,-23097,14907,10146,14869,1245,14829,16181,14780,29764,14741,14732,14701,13401,14652,17261,14613,-20836,14572,-26827,14523,-23822,14484,31057,14443,30754,14394,-24729,14355,-21239,14313,-6645,14264,17481,14226,23413,14183,-4464,14134,-25578,14096,-612,-16513,-763,-16513,-951,-16513,-1184,-16513,-1474,-16513,-1833,-16513,-2279,-16513,-2833,-16513,-3519,-16513,-4368,-16513,-5420,-16513,-6720,-16513,-8328,-16513,-10313,-16513,-12763,-16513,-15785,-16513,-19508,-16513,-24091,-16513,-29728,-16513,28882,-16513,20380,-16513,9954,-16513,-2817,-16514,-18442,-16514,27999,-16514,4694,-16514,-23710,-16515,7255,-16515,30778,-16516,-20196,-16517,-16401,-16518,-25511,-16519,15491,-16520,-27422,-16522,-26629,-16524,13868,-16526,23923,-16529,-1728,-16533,-3499,-16537,11971,-16541,-28148,-16547,-631,-16554,20779,-16561,27824,-16570,12686,-16580,-31315,-16592,22326,-16605,-24170,-16620,29649,-16636,-4153,-16667,-2732,-16704,-6708,-16742,5229,-16790,861,-16862,-26396,-16958,-24451,-17106,21767,-17532,-5593,15517,-12654,15589,12414,15569,26582,15498,-23290,15355,-11221,15082,0,-32768,0,-32768,-11221,-17686,-23290,-17413,26582,-17270,12414,-17199,-12654,-17179,-5593,-17251,21767,15236,-24451,15662,-26396,15810,861,15906,5229,15978,-6708,16026,-2732,16064,-4153,16101,29649,16132,-24170,16148,22326,16163,-31315,16176,12686,16188,27824,16198,20779,16207,-631,16214,-28148,16221,11971,16227,-3499,16231,-1728,16235,23923,16239,13868,16242,-26629,16244,-27422,16246,15491,16248,-25511,16249,-16401,16250,-20196,16251,30778,16252,7255,16253,-23710,16253,4694,16254,27999,16254,-18442,16254,-2817,16254,9954,16255,20380,16255,28882,16255,-29728,16255,-24091,16255,-19508,16255,-15785,16255,-12763,16255,-10313,16255,-8328,16255,-6720,16255,-5420,16255,-4368,16255,-3519,16255,-2833,16255,-2279,16255,-1833,16255,-1474,16255,-1184,16255,-951,16255,-763,16255,-612,16255
    };*/
    // uint32_t NonLinearsLUT[514+254]={0, 524117, 1047213, 1568272, 2086297,
    // 2600313, 3109375, 3612576, 4109053, 4597990, 5078626, 5550257, 6012239,
    // 6463992, 6904999, 7334810, 7753039, 8159364, 8553527, 8935331, 9304639,
    // 9661367, 10005487, 10337020, 10656031, 10962628, 11256959, 11539204,
    // 11809576, 12068312, 12315676, 12551948, 12777429, 12992429, 13197273,
    // 13392290, 13577818, 13754196, 13921765, 14080866, 14231837, 14375013,
    // 14510724, 14639295, 14761042, 14876275, 14985297, 15088399, 15185867,
    // 15277974, 15364985, 15447156, 15524733, 15597950, 15667034, 15732202,
    // 15793660, 15851607, 15906231, 15957712, 16006222, 16051925, 16094974,
    // 16135519, 16173698, 16209645, 16243485, 16275338, 16305318, 16333531,
    // 16360078, 16385056, 16408554, 16430660, 16451453, 16471009, 16489402,
    // 16506699, 16522965, 16538260, 16552640, 16566161, 16578872, 16590823,
    // 16602057, 16612617, 16622543, 16631874, 16640644, 16648887, 16656634,
    // 16663915, 16670757, 16677188, 16683231, 16688911, 16694248, 16699263,
    // 16703975, 16708403, 16712565, 16716474, 16720148, 16723600, 16726844,
    // 16729891, 16732755, 16735445, 16737973, 16740348, 16742579, 16744676,
    // 16746645, 16748496, 16750234, 16751868, 16753402, 16754844, 16756199,
    // 16757471, 16758667, 16759790, 16760845, 16761837, 16762768, 16763643,
    // 16764465, 16765237, 16765963, 16766644, 16767284, 16767886, 16768451,
    // 16768982, 16769480, 16769949, 16770389, 16770803, 16771191, 16771556,
    // 16771899, 16772221, 16772523, 16772808, 16773075, 16773325, 16773561,
    // 16773782, 16773990, 16774186, 16774369, 16774542, 16774704, 16774856,
    // 16774999, 16775133, 16775259, 16775378, 16775489, 16775593, 16775692,
    // 16775784, 16775871, 16775952, 16776029, 16776101, 16776168, 16776231,
    // 16776291, 16776347, 16776400, 16776449, 16776495, 16776539, 16776580,
    // 16776618, 16776655, 16776689, 16776720, 16776750, 16776779, 16776805,
    // 16776830, 16776853, 16776875, 16776896, 16776915, 16776933, 16776950,
    // 16776966, 16776981, 16776996, 16777009, 16777021, 16777033, 16777044,
    // 16777054, 16777064, 16777073, 16777082, 16777090, 16777098, 16777105,
    // 16777111, 16777118, 16777124, 16777129, 16777134, 16777139, 16777144,
    // 16777148, 16777152, 16777156, 16777160, 16777163, 16777166, 16777169,
    // 16777172, 16777174, 16777177, 16777179, 16777181, 16777183, 16777185,
    // 16777187, 16777189, 16777190, 16777192, 16777193, 16777195, 16777196,
    // 16777197, 16777198, 16777199, 16777200, 16777201, 16777202, 16777203,
    // 16777203, 16777204, 16777205, 16777205, 16777206, 16777206, 16777207,
    // 16777207, 16777208, 16777208, 16777209, 16777209, 16777210, 16777210,
    // 16777210, 16777210, 16777211, 16777211, 16777211, 16777215, 16760842,
    // 16711849, 16630619, 16517778, 16374191, 16200945, 15999332, 15770831,
    // 15517082, 15239865, 14941073, 14622685, 14286743, 13935324, 13570518,
    // 13194403, 12809022, 12416371, 12018371, 11616865, 11213597, 10810207,
    // 10408221, 10009048, 9613975, 9224166, 8840663, 8464389, 8096146, 7736628,
    // 7386419, 7046000, 6715758, 6395992, 6086916, 5788672, 5501332, 5224908,
    // 4959355, 4704580, 4460451, 4226794, 4003405, 3790056, 3586494, 3392448,
    // 3207633, 3031754, 2864507, 2705584, 2554673, 2411462, 2275640, 2146899,
    // 2024936, 1909450, 1800151, 1696752, 1598976, 1506554, 1419226, 1336738,
    // 1258849, 1185324, 1115940, 1050480, 988740, 930521, 875635, 823903,
    // 775152, 729220, 685952, 645199, 606822, 570688, 536669, 504647, 474508,
    // 446144, 419453, 394340, 370714, 348488, 327581, 307917, 289423, 272031,
    // 255675, 240296, 225835, 212240, 199457, 187441, 176144, 165525, 155543,
    // 146160, 137341, 129052, 121261, 113939, 107058, 100591, 94514, 88803,
    // 83436, 78393, 73654, 69200, 65016, 61084, 57389, 53918, 50656, 47591,
    // 44712, 42006, 39464, 37076, 34832, 32724, 30743, 28882, 27133, 25491,
    // 23947, 22497, 21135, 19855, 18653, 17524, 16462, 15465, 14529, 13649,
    // 12822, 12046, 11316, 10631, 9987, 9382, 8814, 8280, 7778, 7307, 6865,
    // 6449, 6058, 5691, 5346, 5022, 4718, 4432, 4164, 3912, 3675, 3452, 3243,
    // 3046, 2862, 2689, 2526, 2373, 2229, 2094, 1967, 1848, 1736, 1631, 1532,
    // 1439, 1352, 1270, 1193, 1121, 1053, 989, 929, 873, 820, 770, 724, 680,
    // 639, 600, 564, 529, 497, 467, 439, 412, 387, 364, 342, 321, 302, 283,
    // 266, 250, 235, 221, 207, 195, 183, 172, 161, 152, 142, 134, 126, 118,
    // 111, 104, 98, 92, 86, 81, 76, 72, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38,
    // 36, 34, 32, 30, 28, 26, 25, 23, 22, 21, 19, 18, 17, 16, 15, 14, 13, 12,
    // 12, 11, 10, 10, 9, 9, 8, 8,
    // 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    // uint16_t
    // NonLinearsLUT[514*2]={0,0,7,65365,15,64173,23,60944,31,54681,39,44409,47,29183,55,8096,62,45821,70,10470,77,32354,84,45233,91,48463,98,41464,105,23719,111,60314,118,19791,124,32900,130,33847,136,22435,141,64063,147,27575,152,44015,157,47868,162,39199,167,18116,171,50303,176,4868,180,13096,184,9688,187,60444,191,34572,194,63445,198,16301,201,24537,204,22946,207,11866,209,57172,212,28133,214,56162,217,10525,219,22629,221,27268,223,24767,225,15442,226,65139,228,43089,230,15119,231,47051,233,8086,234,29561,235,46196,236,58237,238,382,239,3930,240,3562,240,65020,241,57431,242,46519,243,32464,244,15438,244,61141,245,38654,246,13663,246,51842,247,22253,247,56093,248,22410,248,52390,249,15067,249,41614,250,1056,250,24554,250,46660,251,1917,251,21473,251,39866,251,57163,252,7893,252,23188,252,37568,252,51089,252,63800,253,10215,253,21449,253,32009,253,41935,253,51266,253,60036,254,2743,254,10490,254,17771,254,24613,254,31044,254,37087,254,42767,254,48104,254,53119,254,57831,254,62259,255,885,255,4794,255,8468,255,11920,255,15164,255,18211,255,21075,255,23765,255,26293,255,28668,255,30899,255,32996,255,34965,255,36816,255,38554,255,40188,255,41722,255,43164,255,44519,255,45791,255,46987,255,48110,255,49165,255,50157,255,51088,255,51963,255,52785,255,53557,255,54283,255,54964,255,55604,255,56206,255,56771,255,57302,255,57800,255,58269,255,58709,255,59123,255,59511,255,59876,255,60219,255,60541,255,60843,255,61128,255,61395,255,61645,255,61881,255,62102,255,62310,255,62506,255,62689,255,62862,255,63024,255,63176,255,63319,255,63453,255,63579,255,63698,255,63809,255,63913,255,64012,255,64104,255,64191,255,64272,255,64349,255,64421,255,64488,255,64551,255,64611,255,64667,255,64720,255,64769,255,64815,255,64859,255,64900,255,64938,255,64975,255,65009,255,65040,255,65070,255,65099,255,65125,255,65150,255,65173,255,65195,255,65216,255,65235,255,65253,255,65270,255,65286,255,65301,255,65316,255,65329,255,65341,255,65353,255,65364,255,65374,255,65384,255,65393,255,65402,255,65410,255,65418,255,65425,255,65431,255,65438,255,65444,255,65449,255,65454,255,65459,255,65464,255,65468,255,65472,255,65476,255,65480,255,65483,255,65486,255,65489,255,65492,255,65494,255,65497,255,65499,255,65501,255,65503,255,65505,255,65507,255,65509,255,65510,255,65512,255,65513,255,65515,255,65516,255,65517,255,65518,255,65519,255,65520,255,65521,255,65522,255,65523,255,65523,255,65524,255,65525,255,65525,255,65526,255,65526,255,65527,255,65527,255,65528,255,65528,255,65529,255,65529,255,65530,255,65530,255,65530,255,65530,255,65531,255,65531,255,65535,255,49162,255,169,253,50011,252,2706,249,55727,247,13553,244,8548,240,42191,236,50586,232,35513,227,64401,223,8157,217,65431,212,41692,207,4566,201,21667,195,29502,189,30067,183,25283,177,16993,171,6941,164,62303,158,53533,152,47576,146,45719,140,49126,134,58839,129,10245,123,35218,118,3380,112,46387,107,33648,102,31086,97,39000,92,57604,88,21504,83,61844,79,47564,75,44155,71,51524,68,4003,64,32490,61,5709,57,54504,54,47550,51,50112,48,61905,46,17098,43,46459,41,18608,38,64305,36,52166,34,47416,32,49747,30,58856,29,8906,27,30679,25,58352,24,26112,22,64762,21,42970,20,26018,19,13665,18,5676,17,1828,16,1904,15,5700,14,13017,13,23667,12,37471,11,54256,11,8324,10,30592,9,55375,9,16998,8,46400,8,12381,7,45895,7,15756,6,52928,6,26237,6,1124,5,43034,5,20808,4,65437,4,45773,4,27279,4,9887,3,59067,3,43688,3,29227,3,15632,3,2849,2,56369,2,45072,2,34453,2,24471,2,15088,2,6269,1,63516,1,55725,1,48403,1,41522,1,35055,1,28978,1,23267,1,17900,1,12857,1,8118,1,3664,0,65016,0,61084,0,57389,0,53918,0,50656,0,47591,0,44712,0,42006,0,39464,0,37076,0,34832,0,32724,0,30743,0,28882,0,27133,0,25491,0,23947,0,22497,0,21135,0,19855,0,18653,0,17524,0,16462,0,15465,0,14529,0,13649,0,12822,0,12046,0,11316,0,10631,0,9987,0,9382,0,8814,0,8280,0,7778,0,7307,0,6865,0,6449,0,6058,0,5691,0,5346,0,5022,0,4718,0,4432,0,4164,0,3912,0,3675,0,3452,0,3243,0,3046,0,2862,0,2689,0,2526,0,2373,0,2229,0,2094,0,1967,0,1848,0,1736,0,1631,0,1532,0,1439,0,1352,0,1270,0,1193,0,1121,0,1053,0,989,0,929,0,873,0,820,0,770,0,724,0,680,0,639,0,600,0,564,0,529,0,497,0,467,0,439,0,412,0,387,0,364,0,342,0,321,0,302,0,283,0,266,0,250,0,235,0,221,0,207,0,195,0,183,0,172,0,161,0,152,0,142,0,134,0,126,0,118,0,111,0,104,0,98,0,92,0,86,0,81,0,76,0,72,0,67,0,63,0,59,0,56,0,52,0,49,0,46,0,43,0,41,0,38,0,36,0,34,0,32,0,30,0,28,0,26,0,25,0,23,0,22,0,21,0,19,0,18,0,17,0,16,0,15,0,14,0,13,0,12,0,12,0,11,0,10,0,10,0,9,0,9,0,8};
    // uint32_t NonLinearsLUT[257+255] = {0, 524117, 1047213, 1568272, 2086297,
    // 2600313, 3109375, 3612576, 4109053, 4597990, 5078626, 5550257, 6012239,
    // 6463992, 6904999, 7334810, 7753039, 8159364, 8553527, 8935331, 9304639,
    // 9661367, 10005487, 10337020, 10656031, 10962628, 11256959, 11539204,
    // 11809576, 12068312, 12315676, 12551948, 12777429, 12992429, 13197273,
    // 13392290, 13577818, 13754196, 13921765, 14080866, 14231837, 14375013,
    // 14510724, 14639295, 14761042, 14876275, 14985297, 15088399, 15185867,
    // 15277974, 15364985, 15447156, 15524733, 15597950, 15667034, 15732202,
    // 15793660, 15851607, 15906231, 15957712, 16006222, 16051925, 16094974,
    // 16135519, 16173698, 16209645, 16243485, 16275338, 16305318, 16333531,
    // 16360078, 16385056, 16408554, 16430660, 16451453, 16471009, 16489402,
    // 16506699, 16522965, 16538260, 16552640, 16566161, 16578872, 16590823,
    // 16602057, 16612617, 16622543, 16631874, 16640644, 16648887, 16656634,
    // 16663915, 16670757, 16677188, 16683231, 16688911, 16694248, 16699263,
    // 16703975, 16708403, 16712565, 16716474, 16720148, 16723600, 16726844,
    // 16729891, 16732755, 16735445, 16737973, 16740348, 16742579, 16744676,
    // 16746645, 16748496, 16750234, 16751868, 16753402, 16754844, 16756199,
    // 16757471, 16758667, 16759790, 16760845, 16761837, 16762768, 16763643,
    // 16764465, 16765237, 16765963, 16766644, 16767284, 16767886, 16768451,
    // 16768982, 16769480, 16769949, 16770389, 16770803, 16771191, 16771556,
    // 16771899, 16772221, 16772523, 16772808, 16773075, 16773325, 16773561,
    // 16773782, 16773990, 16774186, 16774369, 16774542, 16774704, 16774856,
    // 16774999, 16775133, 16775259, 16775378, 16775489, 16775593, 16775692,
    // 16775784, 16775871, 16775952, 16776029, 16776101, 16776168, 16776231,
    // 16776291, 16776347, 16776400, 16776449, 16776495, 16776539, 16776580,
    // 16776618, 16776655, 16776689, 16776720, 16776750, 16776779, 16776805,
    // 16776830, 16776853, 16776875, 16776896, 16776915, 16776933, 16776950,
    // 16776966, 16776981, 16776996, 16777009, 16777021, 16777033, 16777044,
    // 16777054, 16777064, 16777073, 16777082, 16777090, 16777098, 16777105,
    // 16777111, 16777118, 16777124, 16777129, 16777134, 16777139, 16777144,
    // 16777148, 16777152, 16777156, 16777160, 16777163, 16777166, 16777169,
    // 16777172, 16777174, 16777177, 16777179, 16777181, 16777183, 16777185,
    // 16777187, 16777189, 16777190, 16777192, 16777193, 16777195, 16777196,
    // 16777197, 16777198, 16777199, 16777200, 16777201, 16777202, 16777203,
    // 16777203, 16777204, 16777205, 16777205, 16777206, 16777206, 16777207,
    // 16777207, 16777208, 16777208, 16777209, 16777209, 16777210, 16777210,
    // 16777210, 16777210, 16777211, 16777211,
    // 16777211,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    uint32_t NonLinearsLUT[512] = {
        0,        524117,   1047213,  1568272,  2086297,  2600313,  3109375,
        3612576,  4109053,  4597990,  5078626,  5550257,  6012239,  6463992,
        6904999,  7334810,  7753039,  8159364,  8553527,  8935331,  9304639,
        9661367,  10005487, 10337020, 10656031, 10962628, 11256959, 11539204,
        11809576, 12068312, 12315676, 12551948, 12777429, 12992429, 13197273,
        13392290, 13577818, 13754196, 13921765, 14080866, 14231837, 14375013,
        14510724, 14639295, 14761042, 14876275, 14985297, 15088399, 15185867,
        15277974, 15364985, 15447156, 15524733, 15597950, 15667034, 15732202,
        15793660, 15851607, 15906231, 15957712, 16006222, 16051925, 16094974,
        16135519, 16173698, 16209645, 16243485, 16275338, 16305318, 16333531,
        16360078, 16385056, 16408554, 16430660, 16451453, 16471009, 16489402,
        16506699, 16522965, 16538260, 16552640, 16566161, 16578872, 16590823,
        16602057, 16612617, 16622543, 16631874, 16640644, 16648887, 16656634,
        16663915, 16670757, 16677188, 16683231, 16688911, 16694248, 16699263,
        16703975, 16708403, 16712565, 16716474, 16720148, 16723600, 16726844,
        16729891, 16732755, 16735445, 16737973, 16740348, 16742579, 16744676,
        16746645, 16748496, 16750234, 16751868, 16753402, 16754844, 16756199,
        16757471, 16758667, 16759790, 16760845, 16761837, 16762768, 16763643,
        16764465, 16765237, 16765963, 16766644, 16767284, 16767886, 16768451,
        16768982, 16769480, 16769949, 16770389, 16770803, 16771191, 16771556,
        16771899, 16772221, 16772523, 16772808, 16773075, 16773325, 16773561,
        16773782, 16773990, 16774186, 16774369, 16774542, 16774704, 16774856,
        16774999, 16775133, 16775259, 16775378, 16775489, 16775593, 16775692,
        16775784, 16775871, 16775952, 16776029, 16776101, 16776168, 16776231,
        16776291, 16776347, 16776400, 16776449, 16776495, 16776539, 16776580,
        16776618, 16776655, 16776689, 16776720, 16776750, 16776779, 16776805,
        16776830, 16776853, 16776875, 16776896, 16776915, 16776933, 16776950,
        16776966, 16776981, 16776996, 16777009, 16777021, 16777033, 16777044,
        16777054, 16777064, 16777073, 16777082, 16777090, 16777098, 16777105,
        16777111, 16777118, 16777124, 16777129, 16777134, 16777139, 16777144,
        16777148, 16777152, 16777156, 16777160, 16777163, 16777166, 16777169,
        16777172, 16777174, 16777177, 16777179, 16777181, 16777183, 16777185,
        16777187, 16777189, 16777190, 16777192, 16777193, 16777195, 16777196,
        16777197, 16777198, 16777199, 16777200, 16777201, 16777202, 16777203,
        16777203, 16777204, 16777205, 16777205, 16777206, 16777206, 16777207,
        16777207, 16777208, 16777208, 16777209, 16777209, 16777210, 16777210,
        16777210, 16777210, 16777211, 16777211, 16777215, 16760842, 16711849,
        16630619, 16517778, 16374191, 16200945, 15999332, 15770831, 15517082,
        15239865, 14941073, 14622685, 14286743, 13935324, 13570518, 13194403,
        12809022, 12416371, 12018371, 11616865, 11213597, 10810207, 10408221,
        10009048, 9613975,  9224166,  8840663,  8464389,  8096146,  7736628,
        7386419,  7046000,  6715758,  6395992,  6086916,  5788672,  5501332,
        5224908,  4959355,  4704580,  4460451,  4226794,  4003405,  3790056,
        3586494,  3392448,  3207633,  3031754,  2864507,  2705584,  2554673,
        2411462,  2275640,  2146899,  2024936,  1909450,  1800151,  1696752,
        1598976,  1506554,  1419226,  1336738,  1258849,  1185324,  1115940,
        1050480,  988740,   930521,   875635,   823903,   775152,   729220,
        685952,   645199,   606822,   570688,   536669,   504647,   474508,
        446144,   419453,   394340,   370714,   348488,   327581,   307917,
        289423,   272031,   255675,   240296,   225835,   212240,   199457,
        187441,   176144,   165525,   155543,   146160,   137341,   129052,
        121261,   113939,   107058,   100591,   94514,    88803,    83436,
        78393,    73654,    69200,    65016,    61084,    57389,    53918,
        50656,    47591,    44712,    42006,    39464,    37076,    34832,
        32724,    30743,    28882,    27133,    25491,    23947,    22497,
        21135,    19855,    18653,    17524,    16462,    15465,    14529,
        13649,    12822,    12046,    11316,    10631,    9987,     9382,
        8814,     8280,     7778,     7307,     6865,     6449,     6058,
        5691,     5346,     5022,     4718,     4432,     4164,     3912,
        3675,     3452,     3243,     3046,     2862,     2689,     2526,
        2373,     2229,     2094,     1967,     1848,     1736,     1631,
        1532,     1439,     1352,     1270,     1193,     1121,     1053,
        989,      929,      873,      820,      770,      724,      680,
        639,      600,      564,      529,      497,      467,      439,
        412,      387,      364,      342,      321,      302,      283,
        266,      250,      235,      221,      207,      195,      183,
        172,      161,      152,      142,      134,      126,      118,
        111,      104,      98,       92,       86,       81,       76,
        72,       67,       63,       59,       56,       52,       49,
        46,       43,       41,       38,       36,       34,       32,
        30,       28,       26,       25,       23,       22,       21,
        19,       18,       17,       16,       15,       14,       13,
        12,       12,       11,       10,       10,       9,        9,
        8};

    // uint32_t NonLinearsLUT[288] = {0, 524117, 1047213, 1568272, 2086297,
    // 2600313, 3109375, 3612576, 4109053, 4597990, 5078626, 5550257, 6012239,
    // 6463992, 6904999, 7334810, 7753039, 8159364, 8553527, 8935331, 9304639,
    // 9661367, 10005487, 10337020, 10656031, 10962628, 11256959, 11539204,
    // 11809576, 12068312, 12315676, 12551948, 12777429, 12992429, 13197273,
    // 13392290, 13577818, 13754196, 13921765, 14080866, 14231837, 14375013,
    // 14510724, 14639295, 14761042, 14876275, 14985297, 15088399, 15185867,
    // 15277974, 15364985, 15447156, 15524733, 15597950, 15667034, 15732202,
    // 15793660, 15851607, 15906231, 15957712, 16006222, 16051925, 16094974,
    // 16135519, 16173698, 16209645, 16243485, 16275338, 16305318, 16333531,
    // 16360078, 16385056, 16408554, 16430660, 16451453, 16471009, 16489402,
    // 16506699, 16522965, 16538260, 16552640, 16566161, 16578872, 16590823,
    // 16602057, 16612617, 16622543, 16631874, 16640644, 16648887, 16656634,
    // 16663915, 16670757, 16677188, 16683231, 16688911, 16694248, 16699263,
    // 16703975, 16708403, 16712565, 16716474, 16720148, 16723600, 16726844,
    // 16729891, 16732755, 16735445, 16737973, 16740348, 16742579, 16744676,
    // 16746645, 16748496, 16750234, 16751868, 16753402, 16754844, 16756199,
    // 16757471, 16758667, 16759790, 16760845, 16761837, 16762768, 16763643,
    // 16764465, 16765237, 16765963, 16766644, 16767284, 16767886, 16768451,
    // 16768982, 16769480, 16769949, 16770389, 16770803, 16771191, 16771556,
    // 16771899, 16772221, 16772523, 16772808, 16773075, 16773325, 16773561,
    // 16773782, 16773990, 16774186, 16774369, 16774542, 16774704, 16774856,
    // 16774999, 16775133, 16775259, 16775378, 16775489, 16775593, 16775692,
    // 16775784, 16775871, 16775952, 16776029, 16776101, 16776168, 16776231,
    // 16776291, 16776347, 16776400, 16776449, 16776495, 16776539, 16776580,
    // 16776618, 16776655, 16776689, 16776720, 16776750, 16776779, 16776805,
    // 16776830, 16776853, 16776875, 16776896, 16776915, 16776933, 16776950,
    // 16776966, 16776981, 16776996, 16777009, 16777021, 16777033, 16777044,
    // 16777054, 16777064, 16777073, 16777082, 16777090, 16777098, 16777105,
    // 16777111, 16777118, 16777124, 16777129, 16777134, 16777139, 16777144,
    // 16777148, 16777152, 16777156, 16777160, 16777163, 16777166, 16777169,
    // 16777172, 16777174, 16777177, 16777179, 16777181, 16777183, 16777185,
    // 16777187, 16777189, 16777190, 16777192, 16777193, 16777195, 16777196,
    // 16777197, 16777198, 16777199, 16777200, 16777201, 16777202, 16777203,
    // 16777203, 16777204, 16777205, 16777205, 16777206, 16777206, 16777207,
    // 16777207, 16777208, 16777208, 16777209, 16777209, 16777210, 16777210,
    // 16777210, 16777210, 16777211, 16777211, 16777211, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Extract scale and zero-point values and process them
    for (int i = 0; i < 5; i++) {
      Scale[pkeys[i]] = getAttribute<float>(attr, akeys[i] + "_scale");
      ZeroP[pkeys[i]] = static_cast<int64_t>(
          getAttribute<uint16_t>(attr, akeys[i] + "_zero_point"));
    }

    // Fill 'H' initial values
    for (int i = 0; i < 128; i++) {
      HInit[i] = static_cast<int16_t>(ZeroP["H"]);
    }

    for (int p = 0; p < 2; p++) { // Forward or Reverse pass

      // Pre-Process 'B' Biases
      for (int i = 0; i < 512; i++) {
        BiasX = static_cast<int64_t>(BData[p * 1024 + i]) - ZeroP["B"];
        BiasH = static_cast<int64_t>(BData[p * 1024 + 512 + i]) - ZeroP["B"];

        Bias[i] = BiasX + BiasH;

        SumW[i] = 0;
        SumR[i] = 0;

        RowShift = uint32_t(i * RowLen[0]);
        for (int j = 0; j < RowLen[0]; j++) {
          SumW[i] += static_cast<int64_t>(SrcBasePtr[0][RowShift + j]);
        }

        RowShift = uint32_t(i * RowLen[1]);
        for (int j = 0; j < RowLen[1]; j++) {
          SumR[i] += static_cast<int32_t>(SrcBasePtr[1][RowShift + j]);
        }
      }

      // Pre-Process Weights for LSTM0 Node (CH -> HC)
      // **** here
      int16_t tmp_val = (model_variant == "02") ? 32 : 16;

      if (LSTM0Node) {
        uint16_t *TmpSrc =
            (uint16_t *)malloc(WShape[1] * tmp_val * 4 * sizeof(uint16_t));

        for (int r = 0; r < WShape[1];
             r++) { // rows = 512; and below columns 128 = 4 x tmp_val
          for (int h = 0; h < 4; h++) {
            for (int c = 0; c < tmp_val; c++) {
              TmpSrc[r * WShape[2] + h * tmp_val + c] =
                  SrcBasePtr[0][r * WShape[2] + c * 4 + h];
            }
          }
        }

        memcpy((void *)(SrcBasePtr[0]), (void *)(&TmpSrc[0]),
               (WShape[1] * tmp_val * 4) * sizeof(uint16_t));
      }

      computeParamsAndQCVals(Params, QC1Vals, QC2Vals, QDQParams, Scale, ZeroP,
                             SumW, SumR, Bias, RowLen[0], RowLen[1],
                             static_cast<int>(SeqLen));

      fillWeights(io, dst_ptr, SrcBasePtr, RowLen, L1TxferSize);

      for (int i = 0; i < 8; i++) {
        io.write((size_t)(dst_ptr + i * L1TxferSize + QCValsOffset),
                 (void *)(Zeros), 128 * sizeof(int16_t));
        io.write((size_t)(dst_ptr + i * L1TxferSize + HInitOffset),
                 (void *)(HInit), 128 * sizeof(int16_t));
        io.write((size_t)(dst_ptr + i * L1TxferSize + ZerosOffset),
                 (void *)(Zeros), 192 * sizeof(int16_t));
        io.write((size_t)(dst_ptr + i * L1TxferSize + ParamsOffset),
                 (void *)(Params), 64 * sizeof(int16_t));
        io.write((size_t)(dst_ptr + i * L1TxferSize + LUTOffset),
                 (void *)(NonLinearsLUT), 1024 * sizeof(int16_t));
        io.write((size_t)(dst_ptr + i * L1TxferSize + QC1ValsOffset),
                 (void *)(QC1Vals + i * 256), 256 * sizeof(int16_t));
        // std::cout << " ptr = " << i*L1TxferSize + QC1ValsOffset << std::endl;

        io.write((size_t)(dst_ptr + i * L1TxferSize + QC2ValsOffset),
                 (void *)(QC2Vals + i * 256), 256 * sizeof(int16_t));
      }

      // Filling layer params
      uint16_t NumIter[2] = {
          static_cast<uint16_t>(SeqLen),
          (static_cast<uint16_t>(RowLen[p] / (FlipSVWidth[p] * 8)))};

      for (int i = 0; i < 2; i++) { // Loop for {0:Normal, 1:Flip}
        uint8_t LayerParams[128] = {0};
        uint8_t WrapperMode =
            (13 + p * 3 + i); // Need to generate Fwd:{13, 14}, Rev:{16, 17}
        uint16_t CopySize = ((i == 0) ? 128 : 64); // Interms of Bytes

        // No.of Interations, Sequence length, Wrapper Mode, W0 offset, R0
        // offset, W1 offset, R1 offset, Core comute params offset, Width of
        // flip
        packLayerParams(
            LayerParams, QDQParams, NumIter[i], WrapperMode,
            static_cast<uint16_t>(SeqLen),
            // 0, WSize*2*2, WSize*2, (WSize*2 + RSize)*2,
            // QCValsOffset*2, // Offsets are in terms of bytes
            uint16_t(0), uint16_t(WSize * 2), uint16_t(WRSize * 2),
            uint16_t((WRSize + WSize) * 2),
            uint16_t(HInitOffset * 2), // Offsets are in terms of bytes
            ((p == 1 && i == 1) ? FlipSVWidth[1] : FlipSVWidth[0]),
            static_cast<uint16_t>(RowLen[0]), static_cast<uint16_t>(RowLen[1]),
            (LSTM0Node ? 0 : 1), (i == 1), seqlen_iter[p]);
        io.write((size_t)(dst_ptr + L1TxferSize * 8 + i * 128),
                 (void *)(LayerParams), CopySize * sizeof(uint8_t));
        io.write((size_t)(dst_ptr + L1TxferSize * 8 + i * 128 + CopySize / 2),
                 (void *)(LayerParams), CopySize * sizeof(uint8_t));
      }

      // Prepare for next pass
      SrcBasePtr[0] += 512 * RowLen[0];
      SrcBasePtr[1] += 512 * RowLen[1];

      dst_ptr = dst_ptr + L1TxferSize * 8 + LayerParamsSize;
    }

    return;
  }
};

/*        template <typename InT, typename WtT, typename OutT>
        void lstm<InT, WtT, OutT>::initialize_const_params (
            void *dest, const std::vector<Tensor> &const_params,
            const std::map<std::string, std::any> &attr) {

            RYZENAI_LOG_TRACE("LSTM initialize_const_params(ptr) ...");

            LSTMUtil lstmUtil = new();

            lstmUitl.generateL1InitData(dest, const_params, attr);

            RYZENAI_LOG_TRACE("LSTM initialize_const_params(ptr) ... DONE");
        }*/
