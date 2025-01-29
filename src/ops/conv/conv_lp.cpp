// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//

#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <ops/conv/conv_lp.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace conv_lp {
struct mswbjvw_08_160_Layer1 : TilingInfo {
  mswbjvw_08_160_Layer1() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {3, 1, 3, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 64, 160};
    this->align_ofm_shape = {16, 64, 160};
    this->align_ifm = {3, 60, 160};
    this->ofmsv_chunk = {8, 16, 16};
    this->ifmsv_chunk = {8, 18, 24};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_160_Layer2 : TilingInfo {
  mswbjvw_08_160_Layer2() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 32, 80};
    this->align_ofm_shape = {32, 32, 80};
    this->align_ifm = {16, 32, 80};
    this->ofmsv_chunk = {16, 8, 16};
    this->ifmsv_chunk = {16, 10, 24};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_160_Layer3 : TilingInfo {
  mswbjvw_08_160_Layer3() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 40};
    this->align_ofm_shape = {16, 16, 40};
    this->align_ifm = {32, 16, 40};
    this->ofmsv_chunk = {8, 4, 8};
    this->ifmsv_chunk = {32, 4, 8};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer4 : TilingInfo {
  mswbjvw_08_160_Layer4() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29150;
    this->ofm_shape = {32, 16, 40};
    this->align_ofm_shape = {32, 16, 40};
    this->align_ifm = {16, 16, 40};
    this->ofmsv_chunk = {16, 4, 8};
    this->ifmsv_chunk = {16, 6, 16};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer5 : TilingInfo {
  mswbjvw_08_160_Layer5() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 40};
    this->align_ofm_shape = {128, 16, 40};
    this->align_ifm = {32, 16, 40};
    this->ofmsv_chunk = {16, 4, 8};
    this->ifmsv_chunk = {32, 4, 8};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer6 : TilingInfo {
  mswbjvw_08_160_Layer6() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 40};
    this->align_ofm_shape = {16, 16, 40};
    this->align_ifm = {128, 16, 40};
    this->ofmsv_chunk = {8, 4, 8};
    this->ifmsv_chunk = {128, 4, 8};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer7 : TilingInfo {
  mswbjvw_08_160_Layer7() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29150;
    this->ofm_shape = {32, 16, 40};
    this->align_ofm_shape = {32, 16, 40};
    this->align_ifm = {16, 16, 40};
    this->ofmsv_chunk = {16, 4, 8};
    this->ifmsv_chunk = {16, 6, 16};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer8 : TilingInfo {
  mswbjvw_08_160_Layer8() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 40};
    this->align_ofm_shape = {128, 16, 40};
    this->align_ifm = {32, 16, 40};
    this->ofmsv_chunk = {16, 4, 8};
    this->ifmsv_chunk = {32, 4, 8};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_160_Layer9 : TilingInfo {
  mswbjvw_08_160_Layer9() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 40};
    this->align_ofm_shape = {32, 8, 40};
    this->align_ifm = {128, 8, 40};
    this->ofmsv_chunk = {16, 2, 8};
    this->ifmsv_chunk = {128, 2, 8};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer10 : TilingInfo {
  mswbjvw_08_160_Layer10() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29920;
    this->ofm_shape = {48, 8, 40};
    this->align_ofm_shape = {48, 8, 40};
    this->align_ifm = {32, 8, 40};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 1;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer11 : TilingInfo {
  mswbjvw_08_160_Layer11() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 40};
    this->align_ofm_shape = {256, 8, 40};
    this->align_ifm = {48, 8, 40};
    this->ofmsv_chunk = {16, 2, 8};
    this->ifmsv_chunk = {48, 2, 8};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer12 : TilingInfo {
  mswbjvw_08_160_Layer12() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 40};
    this->align_ofm_shape = {32, 8, 40};
    this->align_ifm = {256, 8, 40};
    this->ofmsv_chunk = {16, 2, 8};
    this->ifmsv_chunk = {64, 2, 8};
    this->width_iter = 5;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer13 : TilingInfo {
  mswbjvw_08_160_Layer13() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29920;
    this->ofm_shape = {48, 8, 40};
    this->align_ofm_shape = {48, 8, 40};
    this->align_ifm = {32, 8, 40};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 1;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer14 : TilingInfo {
  mswbjvw_08_160_Layer14() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 40};
    this->align_ofm_shape = {256, 8, 40};
    this->align_ifm = {48, 8, 40};
    this->ofmsv_chunk = {32, 2, 8};
    this->ifmsv_chunk = {48, 2, 8};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_160_Layer15 : TilingInfo {
  mswbjvw_08_160_Layer15() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 40};
    this->align_ofm_shape = {64, 4, 40};
    this->align_ifm = {256, 4, 40};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer16 : TilingInfo {
  mswbjvw_08_160_Layer16() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 40924;
    this->ofm_shape = {80, 4, 40};
    this->align_ofm_shape = {80, 4, 40};
    this->align_ifm = {64, 4, 40};
    this->ofmsv_chunk = {8, 1, 40};
    this->ifmsv_chunk = {16, 3, 48};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer17 : TilingInfo {
  mswbjvw_08_160_Layer17() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 40};
    this->align_ofm_shape = {512, 4, 40};
    this->align_ifm = {80, 4, 40};
    this->ofmsv_chunk = {32, 1, 8};
    this->ifmsv_chunk = {80, 1, 8};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer18 : TilingInfo {
  mswbjvw_08_160_Layer18() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 40};
    this->align_ofm_shape = {64, 4, 40};
    this->align_ifm = {512, 4, 40};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 1;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer19 : TilingInfo {
  mswbjvw_08_160_Layer19() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 40924;
    this->ofm_shape = {80, 4, 40};
    this->align_ofm_shape = {80, 4, 40};
    this->align_ifm = {64, 4, 40};
    this->ofmsv_chunk = {8, 1, 40};
    this->ifmsv_chunk = {16, 3, 48};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer20 : TilingInfo {
  mswbjvw_08_160_Layer20() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 40};
    this->align_ofm_shape = {512, 4, 40};
    this->align_ifm = {80, 4, 40};
    this->ofmsv_chunk = {32, 1, 8};
    this->ifmsv_chunk = {80, 1, 8};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_160_Layer21 : TilingInfo {
  mswbjvw_08_160_Layer21() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 4, 40};
    this->align_ofm_shape = {16, 4, 40};
    this->align_ifm = {512, 4, 40};
    this->ofmsv_chunk = {8, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 1;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};

/* mswbjvw 08_80 tiling information */
struct mswbjvw_08_80_Layer1 : TilingInfo {
  mswbjvw_08_80_Layer1() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {3, 1, 3, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 64, 80};
    this->align_ofm_shape = {16, 64, 80};
    this->align_ifm = {3, 60, 80};
    this->ofmsv_chunk = {8, 16, 16};
    this->ifmsv_chunk = {8, 18, 24};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_80_Layer2 : TilingInfo {
  mswbjvw_08_80_Layer2() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 32, 40};
    this->align_ofm_shape = {32, 32, 48};
    this->align_ifm = {16, 32, 40};
    this->ofmsv_chunk = {16, 8, 16};
    this->ifmsv_chunk = {16, 10, 24};
    this->width_iter = 3;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_80_Layer3 : TilingInfo {
  mswbjvw_08_80_Layer3() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 20};
    this->align_ofm_shape = {16, 16, 24};
    this->align_ifm = {32, 16, 20};
    this->ofmsv_chunk = {8, 4, 24};
    this->ifmsv_chunk = {32, 4, 24};
    this->width_iter = 1;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer4 : TilingInfo {
  mswbjvw_08_80_Layer4() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29150;
    this->ofm_shape = {32, 16, 20};
    this->align_ofm_shape = {32, 16, 24};
    this->align_ifm = {16, 16, 20};
    this->ofmsv_chunk = {16, 4, 8};
    this->ifmsv_chunk = {16, 6, 16};
    this->width_iter = 3;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer5 : TilingInfo {
  mswbjvw_08_80_Layer5() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 20};
    this->align_ofm_shape = {128, 16, 24};
    this->align_ifm = {32, 16, 20};
    this->ofmsv_chunk = {16, 4, 8};
    this->ifmsv_chunk = {32, 4, 8};
    this->width_iter = 3;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer6 : TilingInfo {
  mswbjvw_08_80_Layer6() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 20};
    this->align_ofm_shape = {16, 16, 24};
    this->align_ifm = {128, 16, 20};
    this->ofmsv_chunk = {8, 4, 24};
    this->ifmsv_chunk = {32, 4, 24};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer7 : TilingInfo {
  mswbjvw_08_80_Layer7() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29150;
    this->ofm_shape = {32, 16, 20};
    this->align_ofm_shape = {32, 16, 24};
    this->align_ifm = {16, 16, 20};
    this->ofmsv_chunk = {16, 4, 8};
    this->ifmsv_chunk = {16, 6, 16};
    this->width_iter = 3;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer8 : TilingInfo {
  mswbjvw_08_80_Layer8() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 20};
    this->align_ofm_shape = {128, 16, 24};
    this->align_ifm = {32, 16, 20};
    this->ofmsv_chunk = {32, 4, 8};
    this->ifmsv_chunk = {32, 4, 8};
    this->width_iter = 3;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_80_Layer9 : TilingInfo {
  mswbjvw_08_80_Layer9() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 20};
    this->align_ofm_shape = {32, 8, 24};
    this->align_ifm = {128, 8, 20};
    this->ofmsv_chunk = {16, 2, 24};
    this->ifmsv_chunk = {64, 2, 24};
    this->width_iter = 1;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer10 : TilingInfo {
  mswbjvw_08_80_Layer10() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29920;
    this->ofm_shape = {48, 8, 20};
    this->align_ofm_shape = {48, 8, 24};
    this->align_ifm = {32, 8, 20};
    this->ofmsv_chunk = {8, 2, 24};
    this->ifmsv_chunk = {32, 4, 32};
    this->width_iter = 1;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer11 : TilingInfo {
  mswbjvw_08_80_Layer11() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 20};
    this->align_ofm_shape = {256, 8, 24};
    this->align_ifm = {48, 8, 20};
    this->ofmsv_chunk = {32, 2, 8};
    this->ifmsv_chunk = {48, 2, 8};
    this->width_iter = 3;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer12 : TilingInfo {
  mswbjvw_08_80_Layer12() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 20};
    this->align_ofm_shape = {32, 8, 24};
    this->align_ifm = {256, 8, 20};
    this->ofmsv_chunk = {16, 2, 24};
    this->ifmsv_chunk = {64, 2, 24};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer13 : TilingInfo {
  mswbjvw_08_80_Layer13() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29920;
    this->ofm_shape = {48, 8, 20};
    this->align_ofm_shape = {48, 8, 24};
    this->align_ifm = {32, 8, 20};
    this->ofmsv_chunk = {8, 2, 24};
    this->ifmsv_chunk = {32, 4, 32};
    this->width_iter = 1;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer14 : TilingInfo {
  mswbjvw_08_80_Layer14() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 20};
    this->align_ofm_shape = {256, 8, 24};
    this->align_ifm = {48, 8, 20};
    this->ofmsv_chunk = {64, 2, 8};
    this->ifmsv_chunk = {48, 2, 8};
    this->width_iter = 3;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_80_Layer15 : TilingInfo {
  mswbjvw_08_80_Layer15() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 20};
    this->align_ofm_shape = {64, 4, 24};
    this->align_ifm = {256, 4, 20};
    this->ofmsv_chunk = {16, 1, 24};
    this->ifmsv_chunk = {64, 1, 24};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer16 : TilingInfo {
  mswbjvw_08_80_Layer16() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 40924;
    this->ofm_shape = {80, 4, 20};
    this->align_ofm_shape = {80, 4, 24};
    this->align_ifm = {64, 4, 20};
    this->ofmsv_chunk = {8, 1, 24};
    this->ifmsv_chunk = {16, 3, 32};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer17 : TilingInfo {
  mswbjvw_08_80_Layer17() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 20};
    this->align_ofm_shape = {512, 4, 24};
    this->align_ifm = {80, 4, 20};
    this->ofmsv_chunk = {32, 1, 8};
    this->ifmsv_chunk = {80, 1, 8};
    this->width_iter = 3;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer18 : TilingInfo {
  mswbjvw_08_80_Layer18() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 20};
    this->align_ofm_shape = {64, 4, 24};
    this->align_ifm = {512, 4, 20};
    this->ofmsv_chunk = {16, 1, 24};
    this->ifmsv_chunk = {64, 1, 24};
    this->width_iter = 1;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer19 : TilingInfo {
  mswbjvw_08_80_Layer19() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 40924;
    this->ofm_shape = {80, 4, 20};
    this->align_ofm_shape = {80, 4, 24};
    this->align_ifm = {64, 4, 20};
    this->ofmsv_chunk = {8, 1, 24};
    this->ifmsv_chunk = {16, 3, 32};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer20 : TilingInfo {
  mswbjvw_08_80_Layer20() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 20};
    this->align_ofm_shape = {512, 4, 24};
    this->align_ifm = {80, 4, 20};
    this->ofmsv_chunk = {32, 1, 8};
    this->ifmsv_chunk = {80, 1, 8};
    this->width_iter = 3;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_80_Layer21 : TilingInfo {
  mswbjvw_08_80_Layer21() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 4, 20};
    this->align_ofm_shape = {16, 4, 24};
    this->align_ifm = {512, 4, 20};
    this->ofmsv_chunk = {8, 1, 24};
    this->ifmsv_chunk = {128, 1, 24};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};

/* mswbjvw 320 tiling information */
struct mswbjvw_320_Layer1 : TilingInfo {
  mswbjvw_320_Layer1() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {3, 1, 3, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 64, 320};
    this->align_ofm_shape = {16, 64, 320};
    this->align_ifm = {8, 60, 320};
    this->ofmsv_chunk = {8, 16, 16};
    this->ifmsv_chunk = {8, 18, 24};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_320_Layer2 : TilingInfo {
  mswbjvw_320_Layer2() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 32, 160};
    this->align_ofm_shape = {32, 32, 160};
    this->align_ifm = {16, 32, 160};
    this->ofmsv_chunk = {16, 8, 16};
    this->ifmsv_chunk = {16, 10, 24};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_320_Layer3 : TilingInfo {
  mswbjvw_320_Layer3() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 80};
    this->align_ofm_shape = {16, 16, 80};
    this->align_ifm = {32, 16, 80};
    this->ofmsv_chunk = {8, 4, 16};
    this->ifmsv_chunk = {32, 4, 16};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer4 : TilingInfo {
  mswbjvw_320_Layer4() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 33366;
    this->ofm_shape = {32, 16, 80};
    this->align_ofm_shape = {32, 16, 80};
    this->align_ifm = {16, 16, 80};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer5 : TilingInfo {
  mswbjvw_320_Layer5() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 80};
    this->align_ofm_shape = {128, 16, 80};
    this->align_ifm = {32, 16, 80};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {32, 4, 16};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 4};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer6 : TilingInfo {
  mswbjvw_320_Layer6() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 80};
    this->align_ofm_shape = {16, 16, 80};
    this->align_ifm = {128, 16, 80};
    this->ofmsv_chunk = {8, 4, 16};
    this->ifmsv_chunk = {32, 4, 16};
    this->width_iter = 5;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer7 : TilingInfo {
  mswbjvw_320_Layer7() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 33366;
    this->ofm_shape = {32, 16, 80};
    this->align_ofm_shape = {32, 16, 80};
    this->align_ifm = {16, 16, 80};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer8 : TilingInfo {
  mswbjvw_320_Layer8() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 80};
    this->align_ofm_shape = {128, 16, 80};
    this->align_ifm = {32, 16, 80};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {32, 4, 16};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 4};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_320_Layer9 : TilingInfo {
  mswbjvw_320_Layer9() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 80};
    this->align_ofm_shape = {32, 8, 80};
    this->align_ifm = {128, 8, 80};
    this->ofmsv_chunk = {16, 2, 16};
    this->ifmsv_chunk = {64, 2, 16};
    this->width_iter = 5;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer10 : TilingInfo {
  mswbjvw_320_Layer10() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 34146;
    this->ofm_shape = {48, 8, 80};
    this->align_ofm_shape = {48, 8, 80};
    this->align_ifm = {32, 8, 80};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 2;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 3};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer11 : TilingInfo {
  mswbjvw_320_Layer11() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 80};
    this->align_ofm_shape = {256, 8, 80};
    this->align_ifm = {48, 8, 80};
    this->ofmsv_chunk = {16, 2, 16};
    this->ifmsv_chunk = {48, 2, 16};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 8};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer12 : TilingInfo {
  mswbjvw_320_Layer12() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 80};
    this->align_ofm_shape = {32, 8, 80};
    this->align_ifm = {256, 8, 80};
    this->ofmsv_chunk = {16, 2, 16};
    this->ifmsv_chunk = {64, 2, 16};
    this->width_iter = 5;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer13 : TilingInfo {
  mswbjvw_320_Layer13() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 34146;
    this->ofm_shape = {48, 8, 80};
    this->align_ofm_shape = {48, 8, 80};
    this->align_ifm = {32, 8, 80};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 2;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 3};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer14 : TilingInfo {
  mswbjvw_320_Layer14() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 80};
    this->align_ofm_shape = {256, 8, 80};
    this->align_ifm = {48, 8, 80};
    this->ofmsv_chunk = {16, 2, 16};
    this->ifmsv_chunk = {48, 2, 16};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 8};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_320_Layer15 : TilingInfo {
  mswbjvw_320_Layer15() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 80};
    this->align_ofm_shape = {64, 4, 80};
    this->align_ifm = {256, 4, 80};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 2;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 2};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer16 : TilingInfo {
  mswbjvw_320_Layer16() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 25067;
    this->ofm_shape = {80, 4, 80};
    this->align_ofm_shape = {80, 4, 80};
    this->align_ifm = {64, 4, 80};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 1;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 5};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer17 : TilingInfo {
  mswbjvw_320_Layer17() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 80};
    this->align_ofm_shape = {512, 4, 80};
    this->align_ifm = {80, 4, 80};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {80, 1, 40};
    this->width_iter = 2;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 16;
    this->super_iter = {1, 16};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer18 : TilingInfo {
  mswbjvw_320_Layer18() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 80};
    this->align_ofm_shape = {64, 4, 80};
    this->align_ifm = {512, 4, 80};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 2;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 2};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer19 : TilingInfo {
  mswbjvw_320_Layer19() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 25067;
    this->ofm_shape = {80, 4, 80};
    this->align_ofm_shape = {80, 4, 80};
    this->align_ifm = {64, 4, 80};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 1;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 5};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer20 : TilingInfo {
  mswbjvw_320_Layer20() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 80};
    this->align_ofm_shape = {512, 4, 80};
    this->align_ifm = {80, 4, 80};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {80, 1, 40};
    this->width_iter = 2;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 16;
    this->super_iter = {1, 16};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_320_Layer21 : TilingInfo {
  mswbjvw_320_Layer21() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 4, 80};
    this->align_ofm_shape = {16, 4, 80};
    this->align_ifm = {512, 4, 80};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {32, 1, 80};
    this->width_iter = 1;
    this->depth_iter = 16;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};

struct mswbjvw_640_Layer1 : TilingInfo {
  mswbjvw_640_Layer1() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {3, 1, 3, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 64, 640};
    this->align_ofm_shape = {16, 64, 640};
    this->align_ifm = {8, 60, 640};
    this->ofmsv_chunk = {8, 16, 16};
    this->ifmsv_chunk = {8, 18, 24};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_640_Layer2 : TilingInfo {
  mswbjvw_640_Layer2() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 32, 320};
    this->align_ofm_shape = {32, 32, 320};
    this->align_ifm = {16, 32, 320};
    this->ofmsv_chunk = {16, 8, 16};
    this->ifmsv_chunk = {16, 10, 24};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_640_Layer3 : TilingInfo {
  mswbjvw_640_Layer3() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 160};
    this->align_ofm_shape = {16, 16, 160};
    this->align_ifm = {32, 16, 160};
    this->ofmsv_chunk = {8, 4, 32};
    this->ifmsv_chunk = {32, 4, 32};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer4 : TilingInfo {
  mswbjvw_640_Layer4() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 33336;
    this->ofm_shape = {32, 16, 160};
    this->align_ofm_shape = {32, 16, 160};
    this->align_ifm = {16, 16, 160};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer5 : TilingInfo {
  mswbjvw_640_Layer5() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 160};
    this->align_ofm_shape = {128, 16, 160};
    this->align_ifm = {32, 16, 160};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {32, 4, 16};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 4};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer6 : TilingInfo {
  mswbjvw_640_Layer6() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 160};
    this->align_ofm_shape = {16, 16, 160};
    this->align_ifm = {128, 16, 160};
    this->ofmsv_chunk = {8, 4, 32};
    this->ifmsv_chunk = {32, 4, 32};
    this->width_iter = 5;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer7 : TilingInfo {
  mswbjvw_640_Layer7() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 33336;
    this->ofm_shape = {32, 16, 160};
    this->align_ofm_shape = {32, 16, 160};
    this->align_ifm = {16, 16, 160};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer8 : TilingInfo {
  mswbjvw_640_Layer8() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 160};
    this->align_ofm_shape = {128, 16, 160};
    this->align_ifm = {32, 16, 160};
    this->ofmsv_chunk = {16, 4, 32};
    this->ifmsv_chunk = {32, 4, 32};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 4};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_640_Layer9 : TilingInfo {
  mswbjvw_640_Layer9() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 160};
    this->align_ofm_shape = {32, 8, 160};
    this->align_ifm = {128, 8, 160};
    this->ofmsv_chunk = {16, 2, 32};
    this->ifmsv_chunk = {64, 2, 32};
    this->width_iter = 5;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer10 : TilingInfo {
  mswbjvw_640_Layer10() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 35443;
    this->ofm_shape = {48, 8, 160};
    this->align_ofm_shape = {48, 8, 160};
    this->align_ifm = {32, 8, 160};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 4;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 3};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer11 : TilingInfo {
  mswbjvw_640_Layer11() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 160};
    this->align_ofm_shape = {256, 8, 160};
    this->align_ifm = {48, 8, 160};
    this->ofmsv_chunk = {16, 2, 32};
    this->ifmsv_chunk = {48, 2, 32};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 8};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer12 : TilingInfo {
  mswbjvw_640_Layer12() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 160};
    this->align_ofm_shape = {32, 8, 160};
    this->align_ifm = {256, 8, 160};
    this->ofmsv_chunk = {16, 2, 32};
    this->ifmsv_chunk = {64, 2, 32};
    this->width_iter = 5;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer13 : TilingInfo {
  mswbjvw_640_Layer13() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 35443;
    this->ofm_shape = {48, 8, 160};
    this->align_ofm_shape = {48, 8, 160};
    this->align_ifm = {32, 8, 160};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 4;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 3};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer14 : TilingInfo {
  mswbjvw_640_Layer14() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 160};
    this->align_ofm_shape = {256, 8, 160};
    this->align_ifm = {48, 8, 160};
    this->ofmsv_chunk = {16, 2, 40};
    this->ifmsv_chunk = {48, 2, 40};
    this->width_iter = 4;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 8};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_640_Layer15 : TilingInfo {
  mswbjvw_640_Layer15() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 160};
    this->align_ofm_shape = {64, 4, 160};
    this->align_ifm = {256, 4, 160};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 4;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 2};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer16 : TilingInfo {
  mswbjvw_640_Layer16() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 26894;
    this->ofm_shape = {80, 4, 160};
    this->align_ofm_shape = {80, 4, 160};
    this->align_ifm = {64, 4, 160};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 2;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 5};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer17 : TilingInfo {
  mswbjvw_640_Layer17() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 160};
    this->align_ofm_shape = {512, 4, 160};
    this->align_ifm = {80, 4, 160};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {80, 1, 40};
    this->width_iter = 4;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 16;
    this->super_iter = {1, 16};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer18 : TilingInfo {
  mswbjvw_640_Layer18() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 160};
    this->align_ofm_shape = {64, 4, 160};
    this->align_ifm = {512, 4, 160};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 4;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 2};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer19 : TilingInfo {
  mswbjvw_640_Layer19() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 26894;
    this->ofm_shape = {80, 4, 160};
    this->align_ofm_shape = {80, 4, 160};
    this->align_ifm = {64, 4, 160};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 2;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 5};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer20 : TilingInfo {
  mswbjvw_640_Layer20() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 160};
    this->align_ofm_shape = {512, 4, 160};
    this->align_ifm = {80, 4, 160};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {80, 1, 40};
    this->width_iter = 4;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 16;
    this->super_iter = {1, 16};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_640_Layer21 : TilingInfo {
  mswbjvw_640_Layer21() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 4, 160};
    this->align_ofm_shape = {16, 4, 160};
    this->align_ifm = {512, 4, 160};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {32, 1, 80};
    this->width_iter = 2;
    this->depth_iter = 16;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};

struct mswbjvw_08_1280_Layer1 : TilingInfo {
  mswbjvw_08_1280_Layer1() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {3, 1, 3, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 64, 1280};
    this->align_ofm_shape = {16, 64, 1280};
    this->align_ifm = {3, 60, 1280};
    this->ofmsv_chunk = {8, 16, 16};
    this->ifmsv_chunk = {8, 18, 24};
    this->width_iter = 80;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_1280_Layer2 : TilingInfo {
  mswbjvw_08_1280_Layer2() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 32, 640};
    this->align_ofm_shape = {32, 32, 640};
    this->align_ifm = {16, 32, 640};
    this->ofmsv_chunk = {16, 8, 16};
    this->ifmsv_chunk = {16, 10, 24};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_1280_Layer3 : TilingInfo {
  mswbjvw_08_1280_Layer3() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 320};
    this->align_ofm_shape = {16, 16, 320};
    this->align_ifm = {32, 16, 320};
    this->ofmsv_chunk = {8, 4, 32};
    this->ifmsv_chunk = {32, 4, 32};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer4 : TilingInfo {
  mswbjvw_08_1280_Layer4() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 28755;
    this->ofm_shape = {32, 16, 320};
    this->align_ofm_shape = {32, 16, 320};
    this->align_ifm = {16, 16, 320};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer5 : TilingInfo {
  mswbjvw_08_1280_Layer5() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 320};
    this->align_ofm_shape = {128, 16, 320};
    this->align_ifm = {32, 16, 320};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {32, 4, 16};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer6 : TilingInfo {
  mswbjvw_08_1280_Layer6() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 320};
    this->align_ofm_shape = {16, 16, 320};
    this->align_ifm = {128, 16, 320};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {32, 1, 80};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 16;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer7 : TilingInfo {
  mswbjvw_08_1280_Layer7() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 28755;
    this->ofm_shape = {32, 16, 320};
    this->align_ofm_shape = {32, 16, 320};
    this->align_ifm = {16, 16, 320};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer8 : TilingInfo {
  mswbjvw_08_1280_Layer8() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 320};
    this->align_ofm_shape = {128, 16, 320};
    this->align_ifm = {32, 16, 320};
    this->ofmsv_chunk = {16, 4, 32};
    this->ifmsv_chunk = {32, 4, 32};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_1280_Layer9 : TilingInfo {
  mswbjvw_08_1280_Layer9() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 320};
    this->align_ofm_shape = {32, 8, 320};
    this->align_ifm = {128, 8, 320};
    this->ofmsv_chunk = {16, 2, 32};
    this->ifmsv_chunk = {64, 2, 32};
    this->width_iter = 10;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer10 : TilingInfo {
  mswbjvw_08_1280_Layer10() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29920;
    this->ofm_shape = {48, 8, 320};
    this->align_ofm_shape = {48, 8, 320};
    this->align_ifm = {32, 8, 320};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 8;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer11 : TilingInfo {
  mswbjvw_08_1280_Layer11() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 320};
    this->align_ofm_shape = {256, 8, 320};
    this->align_ifm = {48, 8, 320};
    this->ofmsv_chunk = {16, 2, 32};
    this->ifmsv_chunk = {48, 2, 32};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer12 : TilingInfo {
  mswbjvw_08_1280_Layer12() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 320};
    this->align_ofm_shape = {32, 8, 320};
    this->align_ifm = {256, 8, 320};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 2;
    this->depth_iter = 4;
    this->height_iter = 8;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer13 : TilingInfo {
  mswbjvw_08_1280_Layer13() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 29920;
    this->ofm_shape = {48, 8, 320};
    this->align_ofm_shape = {48, 8, 320};
    this->align_ifm = {32, 8, 320};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 8;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer14 : TilingInfo {
  mswbjvw_08_1280_Layer14() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 320};
    this->align_ofm_shape = {256, 8, 320};
    this->align_ifm = {48, 8, 320};
    this->ofmsv_chunk = {16, 2, 40};
    this->ifmsv_chunk = {48, 2, 40};
    this->width_iter = 8;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_1280_Layer15 : TilingInfo {
  mswbjvw_08_1280_Layer15() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 320};
    this->align_ofm_shape = {64, 4, 320};
    this->align_ifm = {256, 4, 320};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 8;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer16 : TilingInfo {
  mswbjvw_08_1280_Layer16() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 38028;
    this->ofm_shape = {80, 4, 320};
    this->align_ofm_shape = {80, 4, 320};
    this->align_ifm = {64, 4, 320};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 4;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer17 : TilingInfo {
  mswbjvw_08_1280_Layer17() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 320};
    this->align_ofm_shape = {512, 4, 320};
    this->align_ifm = {80, 4, 320};
    this->ofmsv_chunk = {32, 1, 64};
    this->ifmsv_chunk = {80, 1, 64};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer18 : TilingInfo {
  mswbjvw_08_1280_Layer18() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 320};
    this->align_ofm_shape = {64, 4, 320};
    this->align_ifm = {512, 4, 320};
    this->ofmsv_chunk = {32, 1, 16};
    this->ifmsv_chunk = {128, 1, 16};
    this->width_iter = 5;
    this->depth_iter = 4;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer19 : TilingInfo {
  mswbjvw_08_1280_Layer19() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 38028;
    this->ofm_shape = {80, 4, 320};
    this->align_ofm_shape = {80, 4, 320};
    this->align_ifm = {64, 4, 320};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 4;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer20 : TilingInfo {
  mswbjvw_08_1280_Layer20() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 320};
    this->align_ofm_shape = {512, 4, 320};
    this->align_ifm = {80, 4, 320};
    this->ofmsv_chunk = {32, 1, 64};
    this->ifmsv_chunk = {80, 1, 64};
    this->width_iter = 5;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_1280_Layer21 : TilingInfo {
  mswbjvw_08_1280_Layer21() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 4, 320};
    this->align_ofm_shape = {16, 4, 320};
    this->align_ifm = {512, 4, 320};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {32, 1, 80};
    this->width_iter = 1;
    this->depth_iter = 16;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};

struct mswbjvw_2560_Layer1 : TilingInfo {
  mswbjvw_2560_Layer1() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {3, 1, 3, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 64, 2560};
    this->align_ofm_shape = {16, 64, 2560};
    this->align_ifm = {3, 60, 2560};
    this->ofmsv_chunk = {8, 16, 16};
    this->ifmsv_chunk = {8, 18, 24};
    this->width_iter = 160;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_2560_Layer2 : TilingInfo {
  mswbjvw_2560_Layer2() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 32, 1280};
    this->align_ofm_shape = {32, 32, 1280};
    this->align_ifm = {16, 32, 1280};
    this->ofmsv_chunk = {16, 8, 16};
    this->ifmsv_chunk = {16, 10, 24};
    this->width_iter = 80;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_2560_Layer3 : TilingInfo {
  mswbjvw_2560_Layer3() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 640};
    this->align_ofm_shape = {16, 16, 640};
    this->align_ifm = {32, 16, 640};
    this->ofmsv_chunk = {8, 1, 160};
    this->ifmsv_chunk = {32, 1, 160};
    this->width_iter = 1;
    this->depth_iter = 1;
    this->height_iter = 16;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer4 : TilingInfo {
  mswbjvw_2560_Layer4() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 30896;
    this->ofm_shape = {32, 16, 640};
    this->align_ofm_shape = {32, 16, 640};
    this->align_ifm = {16, 16, 640};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer5 : TilingInfo {
  mswbjvw_2560_Layer5() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 640};
    this->align_ofm_shape = {128, 16, 640};
    this->align_ifm = {32, 16, 640};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {32, 4, 16};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer6 : TilingInfo {
  mswbjvw_2560_Layer6() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 640};
    this->align_ofm_shape = {16, 16, 640};
    this->align_ifm = {128, 16, 640};
    this->ofmsv_chunk = {8, 1, 40};
    this->ifmsv_chunk = {128, 1, 40};
    this->width_iter = 4;
    this->depth_iter = 1;
    this->height_iter = 16;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer7 : TilingInfo {
  mswbjvw_2560_Layer7() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 30896;
    this->ofm_shape = {32, 16, 640};
    this->align_ofm_shape = {32, 16, 640};
    this->align_ifm = {16, 16, 640};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer8 : TilingInfo {
  mswbjvw_2560_Layer8() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 640};
    this->align_ofm_shape = {128, 16, 640};
    this->align_ifm = {32, 16, 640};
    this->ofmsv_chunk = {16, 4, 32};
    this->ifmsv_chunk = {32, 4, 32};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_2560_Layer9 : TilingInfo {
  mswbjvw_2560_Layer9() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 640};
    this->align_ofm_shape = {32, 8, 640};
    this->align_ifm = {128, 8, 640};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {128, 1, 40};
    this->width_iter = 4;
    this->depth_iter = 1;
    this->height_iter = 8;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer10 : TilingInfo {
  mswbjvw_2560_Layer10() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 30902;
    this->ofm_shape = {48, 8, 640};
    this->align_ofm_shape = {48, 8, 640};
    this->align_ifm = {32, 8, 640};
    this->ofmsv_chunk = {8, 2, 80};
    this->ifmsv_chunk = {8, 4, 88};
    this->width_iter = 8;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer11 : TilingInfo {
  mswbjvw_2560_Layer11() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 640};
    this->align_ofm_shape = {256, 8, 640};
    this->align_ifm = {48, 8, 640};
    this->ofmsv_chunk = {16, 2, 32};
    this->ifmsv_chunk = {48, 2, 32};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer12 : TilingInfo {
  mswbjvw_2560_Layer12() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 640};
    this->align_ofm_shape = {32, 8, 640};
    this->align_ifm = {256, 8, 640};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 4;
    this->depth_iter = 4;
    this->height_iter = 8;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer13 : TilingInfo {
  mswbjvw_2560_Layer13() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 30902;
    this->ofm_shape = {48, 8, 640};
    this->align_ofm_shape = {48, 8, 640};
    this->align_ifm = {32, 8, 640};
    this->ofmsv_chunk = {8, 2, 80};
    this->ifmsv_chunk = {8, 4, 88};
    this->width_iter = 8;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer14 : TilingInfo {
  mswbjvw_2560_Layer14() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 640};
    this->align_ofm_shape = {256, 8, 640};
    this->align_ifm = {48, 8, 640};
    this->ofmsv_chunk = {16, 2, 40};
    this->ifmsv_chunk = {48, 2, 40};
    this->width_iter = 16;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_2560_Layer15 : TilingInfo {
  mswbjvw_2560_Layer15() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 640};
    this->align_ofm_shape = {64, 4, 640};
    this->align_ifm = {256, 4, 640};
    this->ofmsv_chunk = {32, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 4;
    this->depth_iter = 4;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer16 : TilingInfo {
  mswbjvw_2560_Layer16() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 26413;
    this->ofm_shape = {80, 4, 640};
    this->align_ofm_shape = {80, 4, 640};
    this->align_ifm = {64, 4, 640};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 8;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer17 : TilingInfo {
  mswbjvw_2560_Layer17() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 640};
    this->align_ofm_shape = {512, 4, 640};
    this->align_ifm = {80, 4, 640};
    this->ofmsv_chunk = {32, 1, 64};
    this->ifmsv_chunk = {80, 1, 64};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer18 : TilingInfo {
  mswbjvw_2560_Layer18() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 640};
    this->align_ofm_shape = {64, 4, 640};
    this->align_ifm = {512, 4, 640};
    this->ofmsv_chunk = {32, 1, 32};
    this->ifmsv_chunk = {64, 1, 32};
    this->width_iter = 5;
    this->depth_iter = 8;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer19 : TilingInfo {
  mswbjvw_2560_Layer19() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 26413;
    this->ofm_shape = {80, 4, 640};
    this->align_ofm_shape = {80, 4, 640};
    this->align_ifm = {64, 4, 640};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 8;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer20 : TilingInfo {
  mswbjvw_2560_Layer20() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 640};
    this->align_ofm_shape = {512, 4, 640};
    this->align_ifm = {80, 4, 640};
    this->ofmsv_chunk = {32, 1, 64};
    this->ifmsv_chunk = {80, 1, 64};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_2560_Layer21 : TilingInfo {
  mswbjvw_2560_Layer21() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 4, 640};
    this->align_ofm_shape = {16, 4, 640};
    this->align_ifm = {512, 4, 640};
    this->ofmsv_chunk = {8, 1, 160};
    this->ifmsv_chunk = {32, 1, 160};
    this->width_iter = 1;
    this->depth_iter = 16;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};

struct mswbjvw_08_5120_Layer1 : TilingInfo {
  mswbjvw_08_5120_Layer1() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {3, 1, 3, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 64, 5120};
    this->align_ofm_shape = {16, 64, 5120};
    this->align_ifm = {3, 60, 5120};
    this->ofmsv_chunk = {8, 16, 16};
    this->ifmsv_chunk = {8, 18, 24};
    this->width_iter = 80;
    this->depth_iter = 1;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_5120_Layer2 : TilingInfo {
  mswbjvw_08_5120_Layer2() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 32, 2560};
    this->align_ofm_shape = {32, 32, 2560};
    this->align_ifm = {16, 32, 2560};
    this->ofmsv_chunk = {16, 8, 16};
    this->ifmsv_chunk = {16, 10, 24};
    this->width_iter = 160;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_5120_Layer3 : TilingInfo {
  mswbjvw_08_5120_Layer3() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 1280};
    this->align_ofm_shape = {16, 16, 1280};
    this->align_ifm = {32, 16, 1280};
    this->ofmsv_chunk = {8, 1, 160};
    this->ifmsv_chunk = {32, 1, 160};
    this->width_iter = 2;
    this->depth_iter = 1;
    this->height_iter = 16;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer4 : TilingInfo {
  mswbjvw_08_5120_Layer4() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 28308;
    this->ofm_shape = {32, 16, 1280};
    this->align_ofm_shape = {32, 16, 1280};
    this->align_ifm = {16, 16, 1280};
    this->ofmsv_chunk = {16, 4, 32};
    this->ifmsv_chunk = {16, 6, 40};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer5 : TilingInfo {
  mswbjvw_08_5120_Layer5() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 28685;
    this->ofm_shape = {128, 16, 1280};
    this->align_ofm_shape = {128, 16, 1280};
    this->align_ifm = {32, 16, 1280};
    this->ofmsv_chunk = {64, 1, 40};
    this->ifmsv_chunk = {32, 1, 40};
    this->width_iter = 8;
    this->depth_iter = 1;
    this->height_iter = 16;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer6 : TilingInfo {
  mswbjvw_08_5120_Layer6() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 1280};
    this->align_ofm_shape = {16, 16, 1280};
    this->align_ifm = {128, 16, 1280};
    this->ofmsv_chunk = {8, 1, 40};
    this->ifmsv_chunk = {128, 1, 40};
    this->width_iter = 8;
    this->depth_iter = 1;
    this->height_iter = 16;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer7 : TilingInfo {
  mswbjvw_08_5120_Layer7() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 28308;
    this->ofm_shape = {32, 16, 1280};
    this->align_ofm_shape = {32, 16, 1280};
    this->align_ifm = {16, 16, 1280};
    this->ofmsv_chunk = {16, 4, 32};
    this->ifmsv_chunk = {16, 6, 40};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer8 : TilingInfo {
  mswbjvw_08_5120_Layer8() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 32593;
    this->ofm_shape = {128, 16, 1280};
    this->align_ofm_shape = {128, 16, 1280};
    this->align_ifm = {32, 16, 1280};
    this->ofmsv_chunk = {64, 2, 16};
    this->ifmsv_chunk = {32, 2, 16};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 8;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_5120_Layer9 : TilingInfo {
  mswbjvw_08_5120_Layer9() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 1280};
    this->align_ofm_shape = {32, 8, 1280};
    this->align_ifm = {128, 8, 1280};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {128, 1, 40};
    this->width_iter = 8;
    this->depth_iter = 1;
    this->height_iter = 8;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer10 : TilingInfo {
  mswbjvw_08_5120_Layer10() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 32614;
    this->ofm_shape = {48, 8, 1280};
    this->align_ofm_shape = {48, 8, 1280};
    this->align_ifm = {32, 8, 1280};
    this->ofmsv_chunk = {24, 2, 32};
    this->ifmsv_chunk = {8, 4, 40};
    this->width_iter = 40;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer11 : TilingInfo {
  mswbjvw_08_5120_Layer11() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 31148;
    this->ofm_shape = {256, 8, 1280};
    this->align_ofm_shape = {256, 8, 1280};
    this->align_ifm = {48, 8, 1280};
    this->ofmsv_chunk = {32, 8, 8};
    this->ifmsv_chunk = {48, 8, 8};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer12 : TilingInfo {
  mswbjvw_08_5120_Layer12() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 1280};
    this->align_ofm_shape = {32, 8, 1280};
    this->align_ifm = {256, 8, 1280};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 8;
    this->depth_iter = 4;
    this->height_iter = 8;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer13 : TilingInfo {
  mswbjvw_08_5120_Layer13() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 32614;
    this->ofm_shape = {48, 8, 1280};
    this->align_ofm_shape = {48, 8, 1280};
    this->align_ifm = {32, 8, 1280};
    this->ofmsv_chunk = {24, 2, 32};
    this->ifmsv_chunk = {8, 4, 40};
    this->width_iter = 40;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer14 : TilingInfo {
  mswbjvw_08_5120_Layer14() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 33026;
    this->ofm_shape = {256, 8, 1280};
    this->align_ofm_shape = {256, 8, 1280};
    this->align_ifm = {48, 8, 1280};
    this->ofmsv_chunk = {32, 8, 8};
    this->ifmsv_chunk = {48, 8, 8};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_5120_Layer15 : TilingInfo {
  mswbjvw_08_5120_Layer15() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 1280};
    this->align_ofm_shape = {64, 4, 1280};
    this->align_ifm = {256, 4, 1280};
    this->ofmsv_chunk = {32, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 8;
    this->depth_iter = 4;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer16 : TilingInfo {
  mswbjvw_08_5120_Layer16() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 38876;
    this->ofm_shape = {80, 4, 1280};
    this->align_ofm_shape = {80, 4, 1280};
    this->align_ifm = {64, 4, 1280};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 16;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer17 : TilingInfo {
  mswbjvw_08_5120_Layer17() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 25478;
    this->ofm_shape = {512, 4, 1280};
    this->align_ofm_shape = {512, 4, 1280};
    this->align_ifm = {80, 4, 1280};
    this->ofmsv_chunk = {32, 1, 32};
    this->ifmsv_chunk = {80, 1, 32};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 4;
    this->channel_iter = 8;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer18 : TilingInfo {
  mswbjvw_08_5120_Layer18() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 1280};
    this->align_ofm_shape = {64, 4, 1280};
    this->align_ifm = {512, 4, 1280};
    this->ofmsv_chunk = {32, 1, 32};
    this->ifmsv_chunk = {64, 1, 32};
    this->width_iter = 10;
    this->depth_iter = 8;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer19 : TilingInfo {
  mswbjvw_08_5120_Layer19() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 38876;
    this->ofm_shape = {80, 4, 1280};
    this->align_ofm_shape = {80, 4, 1280};
    this->align_ifm = {64, 4, 1280};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 16;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer20 : TilingInfo {
  mswbjvw_08_5120_Layer20() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 25478;
    this->ofm_shape = {512, 4, 1280};
    this->align_ofm_shape = {512, 4, 1280};
    this->align_ifm = {80, 4, 1280};
    this->ofmsv_chunk = {32, 1, 32};
    this->ifmsv_chunk = {80, 1, 32};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 4;
    this->channel_iter = 8;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_5120_Layer21 : TilingInfo {
  mswbjvw_08_5120_Layer21() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 4, 1280};
    this->align_ofm_shape = {16, 4, 1280};
    this->align_ifm = {512, 4, 1280};
    this->ofmsv_chunk = {8, 1, 40};
    this->ifmsv_chunk = {32, 1, 40};
    this->width_iter = 8;
    this->depth_iter = 16;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};

struct mswbjvw_08_8000_Layer1 : TilingInfo {
  mswbjvw_08_8000_Layer1() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {3, 1, 3, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 64, 8000};
    this->align_ofm_shape = {16, 64, 8000};
    this->align_ifm = {3, 60, 8000};
    this->ofmsv_chunk = {8, 8, 32};
    this->ifmsv_chunk = {8, 10, 40};
    this->width_iter = 250;
    this->depth_iter = 1;
    this->height_iter = 2;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_8000_Layer2 : TilingInfo {
  mswbjvw_08_8000_Layer2() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 32, 4000};
    this->align_ofm_shape = {32, 32, 4000};
    this->align_ifm = {16, 32, 4000};
    this->ofmsv_chunk = {16, 8, 16};
    this->ifmsv_chunk = {16, 10, 24};
    this->width_iter = 250;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_08_8000_Layer3 : TilingInfo {
  mswbjvw_08_8000_Layer3() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 2000};
    this->align_ofm_shape = {16, 16, 2000};
    this->align_ifm = {32, 16, 2000};
    this->ofmsv_chunk = {8, 4, 40};
    this->ifmsv_chunk = {32, 4, 40};
    this->width_iter = 50;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer4 : TilingInfo {
  mswbjvw_08_8000_Layer4() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 28308;
    this->ofm_shape = {32, 16, 2000};
    this->align_ofm_shape = {32, 16, 2000};
    this->align_ifm = {16, 16, 2000};
    this->ofmsv_chunk = {16, 4, 40};
    this->ifmsv_chunk = {16, 6, 48};
    this->width_iter = 50;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer5 : TilingInfo {
  mswbjvw_08_8000_Layer5() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 2000};
    this->align_ofm_shape = {128, 16, 2000};
    this->align_ifm = {32, 16, 2000};
    this->ofmsv_chunk = {8, 4, 40};
    this->ifmsv_chunk = {32, 4, 40};
    this->width_iter = 50;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer6 : TilingInfo {
  mswbjvw_08_8000_Layer6() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 2000};
    this->align_ofm_shape = {16, 16, 2000};
    this->align_ifm = {128, 16, 2000};
    this->ofmsv_chunk = {8, 1, 40};
    this->ifmsv_chunk = {128, 1, 40};
    this->width_iter = 50;
    this->depth_iter = 1;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer7 : TilingInfo {
  mswbjvw_08_8000_Layer7() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 28308;
    this->ofm_shape = {32, 16, 2000};
    this->align_ofm_shape = {32, 16, 2000};
    this->align_ifm = {16, 16, 2000};
    this->ofmsv_chunk = {16, 4, 40};
    this->ifmsv_chunk = {16, 6, 48};
    this->width_iter = 50;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer8 : TilingInfo {
  mswbjvw_08_8000_Layer8() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {128, 16, 2000};
    this->align_ofm_shape = {128, 16, 2000};
    this->align_ifm = {32, 16, 2000};
    this->ofmsv_chunk = {8, 4, 40};
    this->ifmsv_chunk = {32, 4, 40};
    this->width_iter = 50;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_8000_Layer9 : TilingInfo {
  mswbjvw_08_8000_Layer9() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 2000};
    this->align_ofm_shape = {32, 8, 2000};
    this->align_ifm = {128, 8, 2000};
    this->ofmsv_chunk = {16, 2, 40};
    this->ifmsv_chunk = {32, 2, 40};
    this->width_iter = 50;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer10 : TilingInfo {
  mswbjvw_08_8000_Layer10() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 32614;
    this->ofm_shape = {48, 8, 2000};
    this->align_ofm_shape = {48, 8, 2000};
    this->align_ifm = {32, 8, 2000};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {32, 4, 48};
    this->width_iter = 50;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 3};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer11 : TilingInfo {
  mswbjvw_08_8000_Layer11() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 2000};
    this->align_ofm_shape = {256, 8, 2000};
    this->align_ifm = {48, 8, 2000};
    this->ofmsv_chunk = {32, 2, 8};
    this->ifmsv_chunk = {48, 2, 8};
    this->width_iter = 250;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer12 : TilingInfo {
  mswbjvw_08_8000_Layer12() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 2000};
    this->align_ofm_shape = {32, 8, 2000};
    this->align_ifm = {256, 8, 2000};
    this->ofmsv_chunk = {16, 2, 40};
    this->ifmsv_chunk = {32, 2, 40};
    this->width_iter = 50;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer13 : TilingInfo {
  mswbjvw_08_8000_Layer13() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 32614;
    this->ofm_shape = {48, 8, 2000};
    this->align_ofm_shape = {48, 8, 2000};
    this->align_ifm = {32, 8, 2000};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {32, 4, 48};
    this->width_iter = 50;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 3;
    this->super_iter = {1, 3};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer14 : TilingInfo {
  mswbjvw_08_8000_Layer14() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {256, 8, 2000};
    this->align_ofm_shape = {256, 8, 2000};
    this->align_ifm = {48, 8, 2000};
    this->ofmsv_chunk = {32, 2, 8};
    this->ifmsv_chunk = {48, 2, 8};
    this->width_iter = 250;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {1, 1};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_08_8000_Layer15 : TilingInfo {
  mswbjvw_08_8000_Layer15() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 2000};
    this->align_ofm_shape = {64, 4, 2000};
    this->align_ifm = {256, 4, 2000};
    this->ofmsv_chunk = {32, 1, 80};
    this->ifmsv_chunk = {64, 1, 80};
    this->width_iter = 25;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer16 : TilingInfo {
  mswbjvw_08_8000_Layer16() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 38876;
    this->ofm_shape = {80, 4, 2000};
    this->align_ofm_shape = {80, 4, 2000};
    this->align_ifm = {64, 4, 2000};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {16, 3, 88};
    this->width_iter = 25;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 5};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer17 : TilingInfo {
  mswbjvw_08_8000_Layer17() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 2000};
    this->align_ofm_shape = {512, 4, 2000};
    this->align_ifm = {80, 4, 2000};
    this->ofmsv_chunk = {32, 1, 16};
    this->ifmsv_chunk = {80, 1, 16};
    this->width_iter = 125;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 8};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer18 : TilingInfo {
  mswbjvw_08_8000_Layer18() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 2000};
    this->align_ofm_shape = {64, 4, 2000};
    this->align_ifm = {512, 4, 2000};
    this->ofmsv_chunk = {32, 1, 80};
    this->ifmsv_chunk = {64, 1, 80};
    this->width_iter = 25;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer19 : TilingInfo {
  mswbjvw_08_8000_Layer19() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 38876;
    this->ofm_shape = {80, 4, 2000};
    this->align_ofm_shape = {80, 4, 2000};
    this->align_ifm = {64, 4, 2000};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {16, 3, 88};
    this->width_iter = 25;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 5;
    this->super_iter = {1, 5};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer20 : TilingInfo {
  mswbjvw_08_8000_Layer20() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {512, 4, 2000};
    this->align_ofm_shape = {512, 4, 2000};
    this->align_ifm = {80, 4, 2000};
    this->ofmsv_chunk = {32, 1, 16};
    this->ifmsv_chunk = {80, 1, 16};
    this->width_iter = 125;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {1, 8};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_08_8000_Layer21 : TilingInfo {
  mswbjvw_08_8000_Layer21() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 4, 2000};
    this->align_ofm_shape = {16, 4, 2000};
    this->align_ifm = {512, 4, 2000};
    this->ofmsv_chunk = {8, 1, 40};
    this->ifmsv_chunk = {32, 1, 40};
    this->width_iter = 50;
    this->depth_iter = 16;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {1, 1};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};

struct mswbjvw_06_1280_Layer1 : TilingInfo {
  mswbjvw_06_1280_Layer1() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {3, 1, 3, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 64, 1280};
    this->align_ofm_shape = {16, 64, 1280};
    this->align_ifm = {3, 60, 1280};
    this->ofmsv_chunk = {8, 16, 16};
    this->ifmsv_chunk = {8, 18, 24};
    this->width_iter = 80;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_06_1280_Layer2 : TilingInfo {
  mswbjvw_06_1280_Layer2() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 32, 640};
    this->align_ofm_shape = {32, 32, 640};
    this->align_ifm = {16, 32, 640};
    this->ofmsv_chunk = {16, 8, 16};
    this->ifmsv_chunk = {16, 10, 24};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {2, 2};
    this->pool_strides = {2, 2};
  }
};
struct mswbjvw_06_1280_Layer3 : TilingInfo {
  mswbjvw_06_1280_Layer3() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 320};
    this->align_ofm_shape = {16, 16, 320};
    this->align_ifm = {32, 16, 320};
    this->ofmsv_chunk = {8, 4, 32};
    this->ifmsv_chunk = {32, 4, 32};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer4 : TilingInfo {
  mswbjvw_06_1280_Layer4() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 33593;
    this->ofm_shape = {16, 16, 320};
    this->align_ofm_shape = {16, 16, 320};
    this->align_ifm = {16, 16, 320};
    this->ofmsv_chunk = {8, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer5 : TilingInfo {
  mswbjvw_06_1280_Layer5() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 31304;
    this->ofm_shape = {128, 16, 320};
    this->align_ofm_shape = {128, 16, 320};
    this->align_ifm = {16, 16, 320};
    this->ofmsv_chunk = {16, 4, 16};
    this->ifmsv_chunk = {16, 4, 16};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer6 : TilingInfo {
  mswbjvw_06_1280_Layer6() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 16, 320};
    this->align_ofm_shape = {16, 16, 320};
    this->align_ifm = {128, 16, 320};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {32, 1, 80};
    this->width_iter = 1;
    this->depth_iter = 4;
    this->height_iter = 16;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer7 : TilingInfo {
  mswbjvw_06_1280_Layer7() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 33593;
    this->ofm_shape = {16, 16, 320};
    this->align_ofm_shape = {16, 16, 320};
    this->align_ifm = {16, 16, 320};
    this->ofmsv_chunk = {8, 4, 16};
    this->ifmsv_chunk = {16, 6, 24};
    this->width_iter = 20;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer8 : TilingInfo {
  mswbjvw_06_1280_Layer8() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 28705;
    this->ofm_shape = {128, 16, 320};
    this->align_ofm_shape = {128, 16, 320};
    this->align_ifm = {16, 16, 320};
    this->ofmsv_chunk = {16, 4, 32};
    this->ifmsv_chunk = {16, 4, 32};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_06_1280_Layer9 : TilingInfo {
  mswbjvw_06_1280_Layer9() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 320};
    this->align_ofm_shape = {32, 8, 320};
    this->align_ifm = {128, 8, 320};
    this->ofmsv_chunk = {16, 2, 32};
    this->ifmsv_chunk = {64, 2, 32};
    this->width_iter = 10;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer10 : TilingInfo {
  mswbjvw_06_1280_Layer10() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 31882;
    this->ofm_shape = {32, 8, 320};
    this->align_ofm_shape = {32, 8, 320};
    this->align_ifm = {32, 8, 320};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 8;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer11 : TilingInfo {
  mswbjvw_06_1280_Layer11() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 38226;
    this->ofm_shape = {256, 8, 320};
    this->align_ofm_shape = {256, 8, 320};
    this->align_ifm = {32, 8, 320};
    this->ofmsv_chunk = {16, 2, 32};
    this->ifmsv_chunk = {32, 2, 32};
    this->width_iter = 10;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer12 : TilingInfo {
  mswbjvw_06_1280_Layer12() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {32, 8, 320};
    this->align_ofm_shape = {32, 8, 320};
    this->align_ifm = {256, 8, 320};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 2;
    this->depth_iter = 4;
    this->height_iter = 8;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer13 : TilingInfo {
  mswbjvw_06_1280_Layer13() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 31882;
    this->ofm_shape = {32, 8, 320};
    this->align_ofm_shape = {32, 8, 320};
    this->align_ifm = {32, 8, 320};
    this->ofmsv_chunk = {8, 2, 40};
    this->ifmsv_chunk = {16, 4, 48};
    this->width_iter = 8;
    this->depth_iter = 2;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer14 : TilingInfo {
  mswbjvw_06_1280_Layer14() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 36712;
    this->ofm_shape = {256, 8, 320};
    this->align_ofm_shape = {256, 8, 320};
    this->align_ifm = {32, 8, 320};
    this->ofmsv_chunk = {16, 2, 40};
    this->ifmsv_chunk = {32, 2, 40};
    this->width_iter = 8;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {};
    this->pool_ksize = {2, 1};
    this->pool_strides = {2, 1};
  }
};
struct mswbjvw_06_1280_Layer15 : TilingInfo {
  mswbjvw_06_1280_Layer15() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 320};
    this->align_ofm_shape = {64, 4, 320};
    this->align_ifm = {256, 4, 320};
    this->ofmsv_chunk = {16, 1, 40};
    this->ifmsv_chunk = {64, 1, 40};
    this->width_iter = 8;
    this->depth_iter = 4;
    this->height_iter = 1;
    this->channel_iter = 2;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer16 : TilingInfo {
  mswbjvw_06_1280_Layer16() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 34335;
    this->ofm_shape = {64, 4, 320};
    this->align_ofm_shape = {64, 4, 320};
    this->align_ifm = {64, 4, 320};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 4;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer17 : TilingInfo {
  mswbjvw_06_1280_Layer17() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 33814;
    this->ofm_shape = {512, 4, 320};
    this->align_ofm_shape = {512, 4, 320};
    this->align_ifm = {64, 4, 320};
    this->ofmsv_chunk = {32, 1, 8};
    this->ifmsv_chunk = {64, 1, 8};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer18 : TilingInfo {
  mswbjvw_06_1280_Layer18() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {64, 4, 320};
    this->align_ofm_shape = {64, 4, 320};
    this->align_ifm = {512, 4, 320};
    this->ofmsv_chunk = {32, 1, 16};
    this->ifmsv_chunk = {128, 1, 16};
    this->width_iter = 5;
    this->depth_iter = 4;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer19 : TilingInfo {
  mswbjvw_06_1280_Layer19() {
    this->stride = {1, 1};
    this->kernel_size = 3;
    this->padding = {1, 1, 1, 1};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 34335;
    this->ofm_shape = {64, 4, 320};
    this->align_ofm_shape = {64, 4, 320};
    this->align_ifm = {64, 4, 320};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {8, 3, 88};
    this->width_iter = 4;
    this->depth_iter = 8;
    this->height_iter = 1;
    this->channel_iter = 4;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer20 : TilingInfo {
  mswbjvw_06_1280_Layer20() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 33814;
    this->ofm_shape = {512, 4, 320};
    this->align_ofm_shape = {512, 4, 320};
    this->align_ifm = {64, 4, 320};
    this->ofmsv_chunk = {32, 1, 8};
    this->ifmsv_chunk = {64, 1, 8};
    this->width_iter = 40;
    this->depth_iter = 1;
    this->height_iter = 1;
    this->channel_iter = 8;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};
struct mswbjvw_06_1280_Layer21 : TilingInfo {
  mswbjvw_06_1280_Layer21() {
    this->stride = {1, 1};
    this->kernel_size = 1;
    this->padding = {0, 0, 0, 0};
    this->ifm_type = "uint16";
    this->wgt_type = "uint16";
    this->ofm_type = "uint16";
    this->oh = 1;
    this->ow = 8;
    this->ic = 8;
    this->oc = 8;
    this->pad_value = 0;
    this->ofm_shape = {16, 4, 320};
    this->align_ofm_shape = {16, 4, 320};
    this->align_ifm = {512, 4, 320};
    this->ofmsv_chunk = {8, 1, 80};
    this->ifmsv_chunk = {32, 1, 80};
    this->width_iter = 1;
    this->depth_iter = 16;
    this->height_iter = 4;
    this->channel_iter = 1;
    this->super_iter = {};
    this->pool_ksize = {};
    this->pool_strides = {};
  }
};

std::map<std::string, TilingInfo> TILING_INFO_MAP = {
    {"1_8_60_160_1_16_32_80_16_8_3_3_2_2_2_2", mswbjvw_08_160_Layer1()},
    {"1_16_32_80_1_32_16_40_32_16_3_3_2_2_2_2", mswbjvw_08_160_Layer2()},
    {"1_32_16_40_1_16_16_40_16_32_1_1_0_0_0_0", mswbjvw_08_160_Layer3()},
    {"1_16_16_40_1_32_16_40_32_16_3_3_0_0_0_0", mswbjvw_08_160_Layer4()},
    {"1_32_16_40_1_128_16_40_128_32_1_1_0_0_0_0", mswbjvw_08_160_Layer5()},
    {"1_128_16_40_1_16_16_40_16_128_1_1_0_0_0_0", mswbjvw_08_160_Layer6()},
    {"1_16_16_40_1_32_16_40_32_16_3_3_0_0_0_0", mswbjvw_08_160_Layer7()},
    {"1_32_16_40_1_128_8_40_128_32_1_1_2_1_2_1", mswbjvw_08_160_Layer8()},
    {"1_128_8_40_1_32_8_40_32_128_1_1_0_0_0_0", mswbjvw_08_160_Layer9()},
    {"1_32_8_40_1_48_8_40_48_32_3_3_0_0_0_0", mswbjvw_08_160_Layer10()},
    {"1_48_8_40_1_256_8_40_256_48_1_1_0_0_0_0", mswbjvw_08_160_Layer11()},
    {"1_256_8_40_1_32_8_40_32_256_1_1_0_0_0_0", mswbjvw_08_160_Layer12()},
    {"1_32_8_40_1_48_8_40_48_32_3_3_0_0_0_0", mswbjvw_08_160_Layer13()},
    {"1_48_8_40_1_256_4_40_256_48_1_1_2_1_2_1", mswbjvw_08_160_Layer14()},
    {"1_256_4_40_1_64_4_40_64_256_1_1_0_0_0_0", mswbjvw_08_160_Layer15()},
    {"1_64_4_40_1_80_4_40_80_64_3_3_0_0_0_0", mswbjvw_08_160_Layer16()},
    {"1_80_4_40_1_512_4_40_512_80_1_1_0_0_0_0", mswbjvw_08_160_Layer17()},
    {"1_512_4_40_1_64_4_40_64_512_1_1_0_0_0_0", mswbjvw_08_160_Layer18()},
    {"1_64_4_40_1_80_4_40_80_64_3_3_0_0_0_0", mswbjvw_08_160_Layer19()},
    {"1_80_4_40_1_512_4_40_512_80_1_1_0_0_0_0", mswbjvw_08_160_Layer20()},
    {"1_512_4_40_1_16_4_40_16_512_1_1_0_0_0_0", mswbjvw_08_160_Layer21()},

    {"1_8_60_80_1_16_32_40_16_8_3_3_2_2_2_2", mswbjvw_08_80_Layer1()},
    {"1_16_32_40_1_32_16_20_32_16_3_3_2_2_2_2", mswbjvw_08_80_Layer2()},
    {"1_32_16_20_1_16_16_20_16_32_1_1_0_0_0_0", mswbjvw_08_80_Layer3()},
    {"1_16_16_20_1_32_16_20_32_16_3_3_0_0_0_0", mswbjvw_08_80_Layer4()},
    {"1_32_16_20_1_128_16_20_128_32_1_1_0_0_0_0", mswbjvw_08_80_Layer5()},
    {"1_128_16_20_1_16_16_20_16_128_1_1_0_0_0_0", mswbjvw_08_80_Layer6()},
    {"1_16_16_20_1_32_16_20_32_16_3_3_0_0_0_0", mswbjvw_08_80_Layer7()},
    {"1_32_16_20_1_128_8_20_128_32_1_1_2_1_2_1", mswbjvw_08_80_Layer8()},
    {"1_128_8_20_1_32_8_20_32_128_1_1_0_0_0_0", mswbjvw_08_80_Layer9()},
    {"1_32_8_20_1_48_8_20_48_32_3_3_0_0_0_0", mswbjvw_08_80_Layer10()},
    {"1_48_8_20_1_256_8_20_256_48_1_1_0_0_0_0", mswbjvw_08_80_Layer11()},
    {"1_256_8_20_1_32_8_20_32_256_1_1_0_0_0_0", mswbjvw_08_80_Layer12()},
    {"1_32_8_20_1_48_8_20_48_32_3_3_0_0_0_0", mswbjvw_08_80_Layer13()},
    {"1_48_8_20_1_256_4_20_256_48_1_1_2_1_2_1", mswbjvw_08_80_Layer14()},
    {"1_256_4_20_1_64_4_20_64_256_1_1_0_0_0_0", mswbjvw_08_80_Layer15()},
    {"1_64_4_20_1_80_4_20_80_64_3_3_0_0_0_0", mswbjvw_08_80_Layer16()},
    {"1_80_4_20_1_512_4_20_512_80_1_1_0_0_0_0", mswbjvw_08_80_Layer17()},
    {"1_512_4_20_1_64_4_20_64_512_1_1_0_0_0_0", mswbjvw_08_80_Layer18()},
    {"1_64_4_20_1_80_4_20_80_64_3_3_0_0_0_0", mswbjvw_08_80_Layer19()},
    {"1_80_4_20_1_512_4_20_512_80_1_1_0_0_0_0", mswbjvw_08_80_Layer20()},
    {"1_512_4_20_1_16_4_20_16_512_1_1_0_0_0_0", mswbjvw_08_80_Layer21()},

    {"1_8_60_320_1_16_32_160_16_8_3_3_2_2_2_2", mswbjvw_320_Layer1()},
    {"1_16_32_160_1_32_16_80_32_16_3_3_2_2_2_2", mswbjvw_320_Layer2()},
    {"1_32_16_80_1_16_16_80_16_32_1_1_0_0_0_0", mswbjvw_320_Layer3()},
    {"1_16_16_80_1_32_16_80_32_16_3_3_0_0_0_0", mswbjvw_320_Layer4()},
    {"1_32_16_80_1_128_16_80_128_32_1_1_0_0_0_0", mswbjvw_320_Layer5()},
    {"1_128_16_80_1_16_16_80_16_128_1_1_0_0_0_0", mswbjvw_320_Layer6()},
    {"1_16_16_80_1_32_16_80_32_16_3_3_0_0_0_0", mswbjvw_320_Layer7()},
    {"1_32_16_80_1_128_8_80_128_32_1_1_2_1_2_1", mswbjvw_320_Layer8()},
    {"1_128_8_80_1_32_8_80_32_128_1_1_0_0_0_0", mswbjvw_320_Layer9()},
    {"1_32_8_80_1_48_8_80_48_32_3_3_0_0_0_0", mswbjvw_320_Layer10()},
    {"1_48_8_80_1_256_8_80_256_48_1_1_0_0_0_0", mswbjvw_320_Layer11()},
    {"1_256_8_80_1_32_8_80_32_256_1_1_0_0_0_0", mswbjvw_320_Layer12()},
    {"1_32_8_80_1_48_8_80_48_32_3_3_0_0_0_0", mswbjvw_320_Layer13()},
    {"1_48_8_80_1_256_4_80_256_48_1_1_2_1_2_1", mswbjvw_320_Layer14()},
    {"1_256_4_80_1_64_4_80_64_256_1_1_0_0_0_0", mswbjvw_320_Layer15()},
    {"1_64_4_80_1_80_4_80_80_64_3_3_0_0_0_0", mswbjvw_320_Layer16()},
    {"1_80_4_80_1_512_4_80_512_80_1_1_0_0_0_0", mswbjvw_320_Layer17()},
    {"1_512_4_80_1_64_4_80_64_512_1_1_0_0_0_0", mswbjvw_320_Layer18()},
    {"1_64_4_80_1_80_4_80_80_64_3_3_0_0_0_0", mswbjvw_320_Layer19()},
    {"1_80_4_80_1_512_4_80_512_80_1_1_0_0_0_0", mswbjvw_320_Layer20()},
    {"1_512_4_80_1_16_4_80_16_512_1_1_0_0_0_0", mswbjvw_320_Layer21()},

    {"1_8_60_640_1_16_32_320_16_8_3_3_2_2_2_2", mswbjvw_640_Layer1()},
    {"1_16_32_320_1_32_16_160_32_16_3_3_2_2_2_2", mswbjvw_640_Layer2()},
    {"1_32_16_160_1_16_16_160_16_32_1_1_0_0_0_0", mswbjvw_640_Layer3()},
    {"1_16_16_160_1_32_16_160_32_16_3_3_0_0_0_0", mswbjvw_640_Layer4()},
    {"1_32_16_160_1_128_16_160_128_32_1_1_0_0_0_0", mswbjvw_640_Layer5()},
    {"1_128_16_160_1_16_16_160_16_128_1_1_0_0_0_0", mswbjvw_640_Layer6()},
    {"1_16_16_160_1_32_16_160_32_16_3_3_0_0_0_0", mswbjvw_640_Layer7()},
    {"1_32_16_160_1_128_8_160_128_32_1_1_2_1_2_1", mswbjvw_640_Layer8()},
    {"1_128_8_160_1_32_8_160_32_128_1_1_0_0_0_0", mswbjvw_640_Layer9()},
    {"1_32_8_160_1_48_8_160_48_32_3_3_0_0_0_0", mswbjvw_640_Layer10()},
    {"1_48_8_160_1_256_8_160_256_48_1_1_0_0_0_0", mswbjvw_640_Layer11()},
    {"1_256_8_160_1_32_8_160_32_256_1_1_0_0_0_0", mswbjvw_640_Layer12()},
    {"1_32_8_160_1_48_8_160_48_32_3_3_0_0_0_0", mswbjvw_640_Layer13()},
    {"1_48_8_160_1_256_4_160_256_48_1_1_2_1_2_1", mswbjvw_640_Layer14()},
    {"1_256_4_160_1_64_4_160_64_256_1_1_0_0_0_0", mswbjvw_640_Layer15()},
    {"1_64_4_160_1_80_4_160_80_64_3_3_0_0_0_0", mswbjvw_640_Layer16()},
    {"1_80_4_160_1_512_4_160_512_80_1_1_0_0_0_0", mswbjvw_640_Layer17()},
    {"1_512_4_160_1_64_4_160_64_512_1_1_0_0_0_0", mswbjvw_640_Layer18()},
    {"1_64_4_160_1_80_4_160_80_64_3_3_0_0_0_0", mswbjvw_640_Layer19()},
    {"1_80_4_160_1_512_4_160_512_80_1_1_0_0_0_0", mswbjvw_640_Layer20()},
    {"1_512_4_160_1_16_4_160_16_512_1_1_0_0_0_0", mswbjvw_640_Layer21()},

    {"1_8_60_1280_1_16_32_640_16_8_3_3_2_2_2_2", mswbjvw_08_1280_Layer1()},
    {"1_16_32_640_1_32_16_320_32_16_3_3_2_2_2_2", mswbjvw_08_1280_Layer2()},
    {"1_32_16_320_1_16_16_320_16_32_1_1_0_0_0_0", mswbjvw_08_1280_Layer3()},
    {"1_16_16_320_1_32_16_320_32_16_3_3_0_0_0_0", mswbjvw_08_1280_Layer4()},
    {"1_32_16_320_1_128_16_320_128_32_1_1_0_0_0_0", mswbjvw_08_1280_Layer5()},
    {"1_128_16_320_1_16_16_320_16_128_1_1_0_0_0_0", mswbjvw_08_1280_Layer6()},
    {"1_16_16_320_1_32_16_320_32_16_3_3_0_0_0_0", mswbjvw_08_1280_Layer7()},
    {"1_32_16_320_1_128_8_320_128_32_1_1_2_1_2_1", mswbjvw_08_1280_Layer8()},
    {"1_128_8_320_1_32_8_320_32_128_1_1_0_0_0_0", mswbjvw_08_1280_Layer9()},
    {"1_32_8_320_1_48_8_320_48_32_3_3_0_0_0_0", mswbjvw_08_1280_Layer10()},
    {"1_48_8_320_1_256_8_320_256_48_1_1_0_0_0_0", mswbjvw_08_1280_Layer11()},
    {"1_256_8_320_1_32_8_320_32_256_1_1_0_0_0_0", mswbjvw_08_1280_Layer12()},
    {"1_32_8_320_1_48_8_320_48_32_3_3_0_0_0_0", mswbjvw_08_1280_Layer13()},
    {"1_48_8_320_1_256_4_320_256_48_1_1_2_1_2_1", mswbjvw_08_1280_Layer14()},
    {"1_256_4_320_1_64_4_320_64_256_1_1_0_0_0_0", mswbjvw_08_1280_Layer15()},
    {"1_64_4_320_1_80_4_320_80_64_3_3_0_0_0_0", mswbjvw_08_1280_Layer16()},
    {"1_80_4_320_1_512_4_320_512_80_1_1_0_0_0_0", mswbjvw_08_1280_Layer17()},
    {"1_512_4_320_1_64_4_320_64_512_1_1_0_0_0_0", mswbjvw_08_1280_Layer18()},
    {"1_64_4_320_1_80_4_320_80_64_3_3_0_0_0_0", mswbjvw_08_1280_Layer19()},
    {"1_80_4_320_1_512_4_320_512_80_1_1_0_0_0_0", mswbjvw_08_1280_Layer20()},
    {"1_512_4_320_1_16_4_320_16_512_1_1_0_0_0_0", mswbjvw_08_1280_Layer21()},

    {"1_8_60_2560_1_16_32_1280_16_8_3_3_2_2_2_2", mswbjvw_2560_Layer1()},
    {"1_16_32_1280_1_32_16_640_32_16_3_3_2_2_2_2", mswbjvw_2560_Layer2()},
    {"1_32_16_640_1_16_16_640_16_32_1_1_0_0_0_0", mswbjvw_2560_Layer3()},
    {"1_16_16_640_1_32_16_640_32_16_3_3_0_0_0_0", mswbjvw_2560_Layer4()},
    {"1_32_16_640_1_128_16_640_128_32_1_1_0_0_0_0", mswbjvw_2560_Layer5()},
    {"1_128_16_640_1_16_16_640_16_128_1_1_0_0_0_0", mswbjvw_2560_Layer6()},
    {"1_16_16_640_1_32_16_640_32_16_3_3_0_0_0_0", mswbjvw_2560_Layer7()},
    {"1_32_16_640_1_128_8_640_128_32_1_1_2_1_2_1", mswbjvw_2560_Layer8()},
    {"1_128_8_640_1_32_8_640_32_128_1_1_0_0_0_0", mswbjvw_2560_Layer9()},
    {"1_32_8_640_1_48_8_640_48_32_3_3_0_0_0_0", mswbjvw_2560_Layer10()},
    {"1_48_8_640_1_256_8_640_256_48_1_1_0_0_0_0", mswbjvw_2560_Layer11()},
    {"1_256_8_640_1_32_8_640_32_256_1_1_0_0_0_0", mswbjvw_2560_Layer12()},
    {"1_32_8_640_1_48_8_640_48_32_3_3_0_0_0_0", mswbjvw_2560_Layer13()},
    {"1_48_8_640_1_256_4_640_256_48_1_1_2_1_2_1", mswbjvw_2560_Layer14()},
    {"1_256_4_640_1_64_4_640_64_256_1_1_0_0_0_0", mswbjvw_2560_Layer15()},
    {"1_64_4_640_1_80_4_640_80_64_3_3_0_0_0_0", mswbjvw_2560_Layer16()},
    {"1_80_4_640_1_512_4_640_512_80_1_1_0_0_0_0", mswbjvw_2560_Layer17()},
    {"1_512_4_640_1_64_4_640_64_512_1_1_0_0_0_0", mswbjvw_2560_Layer18()},
    {"1_64_4_640_1_80_4_640_80_64_3_3_0_0_0_0", mswbjvw_2560_Layer19()},
    {"1_80_4_640_1_512_4_640_512_80_1_1_0_0_0_0", mswbjvw_2560_Layer20()},
    {"1_512_4_640_1_16_4_640_16_512_1_1_0_0_0_0", mswbjvw_2560_Layer21()},

    {"1_8_60_5120_1_16_32_2560_16_8_3_3_2_2_2_2", mswbjvw_08_5120_Layer1()},
    {"1_16_32_2560_1_32_16_1280_32_16_3_3_2_2_2_2", mswbjvw_08_5120_Layer2()},
    {"1_32_16_1280_1_16_16_1280_16_32_1_1_0_0_0_0", mswbjvw_08_5120_Layer3()},
    {"1_16_16_1280_1_32_16_1280_32_16_3_3_0_0_0_0", mswbjvw_08_5120_Layer4()},
    {"1_32_16_1280_1_128_16_1280_128_32_1_1_0_0_0_0", mswbjvw_08_5120_Layer5()},
    {"1_128_16_1280_1_16_16_1280_16_128_1_1_0_0_0_0", mswbjvw_08_5120_Layer6()},
    {"1_16_16_1280_1_32_16_1280_32_16_3_3_0_0_0_0", mswbjvw_08_5120_Layer7()},
    {"1_32_16_1280_1_128_8_1280_128_32_1_1_2_1_2_1", mswbjvw_08_5120_Layer8()},
    {"1_128_8_1280_1_32_8_1280_32_128_1_1_0_0_0_0", mswbjvw_08_5120_Layer9()},
    {"1_32_8_1280_1_48_8_1280_48_32_3_3_0_0_0_0", mswbjvw_08_5120_Layer10()},
    {"1_48_8_1280_1_256_8_1280_256_48_1_1_0_0_0_0", mswbjvw_08_5120_Layer11()},
    {"1_256_8_1280_1_32_8_1280_32_256_1_1_0_0_0_0", mswbjvw_08_5120_Layer12()},
    {"1_32_8_1280_1_48_8_1280_48_32_3_3_0_0_0_0", mswbjvw_08_5120_Layer13()},
    {"1_48_8_1280_1_256_4_1280_256_48_1_1_2_1_2_1", mswbjvw_08_5120_Layer14()},
    {"1_256_4_1280_1_64_4_1280_64_256_1_1_0_0_0_0", mswbjvw_08_5120_Layer15()},
    {"1_64_4_1280_1_80_4_1280_80_64_3_3_0_0_0_0", mswbjvw_08_5120_Layer16()},
    {"1_80_4_1280_1_512_4_1280_512_80_1_1_0_0_0_0", mswbjvw_08_5120_Layer17()},
    {"1_512_4_1280_1_64_4_1280_64_512_1_1_0_0_0_0", mswbjvw_08_5120_Layer18()},
    {"1_64_4_1280_1_80_4_1280_80_64_3_3_0_0_0_0", mswbjvw_08_5120_Layer19()},
    {"1_80_4_1280_1_512_4_1280_512_80_1_1_0_0_0_0", mswbjvw_08_5120_Layer20()},
    {"1_512_4_1280_1_16_4_1280_16_512_1_1_0_0_0_0", mswbjvw_08_5120_Layer21()},

    {"1_8_60_8000_1_16_32_4000_16_8_3_3_2_2_2_2", mswbjvw_08_8000_Layer1()},
    {"1_16_32_4000_1_32_16_2000_32_16_3_3_2_2_2_2", mswbjvw_08_8000_Layer2()},
    {"1_32_16_2000_1_16_16_2000_16_32_1_1_0_0_0_0", mswbjvw_08_8000_Layer3()},
    {"1_16_16_2000_1_32_16_2000_32_16_3_3_0_0_0_0", mswbjvw_08_8000_Layer4()},
    {"1_32_16_2000_1_128_16_2000_128_32_1_1_0_0_0_0", mswbjvw_08_8000_Layer5()},
    {"1_128_16_2000_1_16_16_2000_16_128_1_1_0_0_0_0", mswbjvw_08_8000_Layer6()},
    {"1_16_16_2000_1_32_16_2000_32_16_3_3_0_0_0_0", mswbjvw_08_8000_Layer7()},
    {"1_32_16_2000_1_128_8_2000_128_32_1_1_2_1_2_1", mswbjvw_08_8000_Layer8()},
    {"1_128_8_2000_1_32_8_2000_32_128_1_1_0_0_0_0", mswbjvw_08_8000_Layer9()},
    {"1_32_8_2000_1_48_8_2000_48_32_3_3_0_0_0_0", mswbjvw_08_8000_Layer10()},
    {"1_48_8_2000_1_256_8_2000_256_48_1_1_0_0_0_0", mswbjvw_08_8000_Layer11()},
    {"1_256_8_2000_1_32_8_2000_32_256_1_1_0_0_0_0", mswbjvw_08_8000_Layer12()},
    {"1_32_8_2000_1_48_8_2000_48_32_3_3_0_0_0_0", mswbjvw_08_8000_Layer13()},
    {"1_48_8_2000_1_256_4_2000_256_48_1_1_2_1_2_1", mswbjvw_08_8000_Layer14()},
    {"1_256_4_2000_1_64_4_2000_64_256_1_1_0_0_0_0", mswbjvw_08_8000_Layer15()},
    {"1_64_4_2000_1_80_4_2000_80_64_3_3_0_0_0_0", mswbjvw_08_8000_Layer16()},
    {"1_80_4_2000_1_512_4_2000_512_80_1_1_0_0_0_0", mswbjvw_08_8000_Layer17()},
    {"1_512_4_2000_1_64_4_2000_64_512_1_1_0_0_0_0", mswbjvw_08_8000_Layer18()},
    {"1_64_4_2000_1_80_4_2000_80_64_3_3_0_0_0_0", mswbjvw_08_8000_Layer19()},
    {"1_80_4_2000_1_512_4_2000_512_80_1_1_0_0_0_0", mswbjvw_08_8000_Layer20()},
    {"1_512_4_2000_1_16_4_2000_16_512_1_1_0_0_0_0", mswbjvw_08_8000_Layer21()},

    {"1_8_60_1280_1_16_32_640_16_8_3_3_2_2_2_2", mswbjvw_06_1280_Layer1()},
    {"1_16_32_640_1_32_16_320_32_16_3_3_2_2_2_2", mswbjvw_06_1280_Layer2()},
    {"1_32_16_320_1_16_16_320_16_32_1_1_0_0_0_0", mswbjvw_06_1280_Layer3()},
    {"1_16_16_320_1_16_16_320_16_16_3_3_0_0_0_0", mswbjvw_06_1280_Layer4()},
    {"1_16_16_320_1_128_16_320_128_16_1_1_0_0_0_0", mswbjvw_06_1280_Layer5()},
    {"1_128_16_320_1_16_16_320_16_128_1_1_0_0_0_0", mswbjvw_06_1280_Layer6()},
    {"1_16_16_320_1_16_16_320_16_16_3_3_0_0_0_0", mswbjvw_06_1280_Layer7()},
    {"1_16_16_320_1_128_8_320_128_16_1_1_2_1_2_1", mswbjvw_06_1280_Layer8()},
    {"1_128_8_320_1_32_8_320_32_128_1_1_0_0_0_0", mswbjvw_06_1280_Layer9()},
    {"1_32_8_320_1_32_8_320_32_32_3_3_0_0_0_0", mswbjvw_06_1280_Layer10()},
    {"1_32_8_320_1_256_8_320_256_32_1_1_0_0_0_0", mswbjvw_06_1280_Layer11()},
    {"1_256_8_320_1_32_8_320_32_256_1_1_0_0_0_0", mswbjvw_06_1280_Layer12()},
    {"1_32_8_320_1_32_8_320_32_32_3_3_0_0_0_0", mswbjvw_06_1280_Layer13()},
    {"1_32_8_320_1_256_4_320_256_32_1_1_2_1_2_1", mswbjvw_06_1280_Layer14()},
    {"1_256_4_320_1_64_4_320_64_256_1_1_0_0_0_0", mswbjvw_06_1280_Layer15()},
    {"1_64_4_320_1_64_4_320_64_64_3_3_0_0_0_0", mswbjvw_06_1280_Layer16()},
    {"1_64_4_320_1_512_4_320_512_64_1_1_0_0_0_0", mswbjvw_06_1280_Layer17()},
    {"1_512_4_320_1_64_4_320_64_512_1_1_0_0_0_0", mswbjvw_06_1280_Layer18()},
    {"1_64_4_320_1_64_4_320_64_64_3_3_0_0_0_0", mswbjvw_06_1280_Layer19()},
    {"1_64_4_320_1_512_4_320_512_64_1_1_0_0_0_0", mswbjvw_06_1280_Layer20()},
    {"1_512_4_320_1_16_4_320_16_512_1_1_0_0_0_0", mswbjvw_06_1280_Layer21()}};

TilingInfo TilingInfo::getInstance(const std::string &key) {
  if (TILING_INFO_MAP.find(key) != TILING_INFO_MAP.end()) {
    return TILING_INFO_MAP[key];
  }
  return {};
}

std::vector<int8_t> get_type_size(const std::vector<std::string> &data_types) {
  std::vector<int8_t> type_res;

  for (const std::string &data_type : data_types) {
    if (data_type == "int8" || data_type == "uint8") {
      type_res.push_back(1);
    } else if (data_type == "int16" || data_type == "uint16") {
      type_res.push_back(2);
    } else if (data_type == "int32" || data_type == "uint32" ||
               data_type == "float32") {
      type_res.push_back(4);
    } else if (data_type == "int64" || data_type == "uint64" ||
               data_type == "float64" || data_type == "double") {
      type_res.push_back(8);
    } else {
      // Handle unrecognized data types (optional)
      std::cerr << "Unrecognized data type: " << data_type << std::endl;
      // Add appropriate error handling or default value
    }
  }

  return type_res;
}

std::vector<std::vector<uint8_t>> computeUtil(bool do_maxpool,
                                              bool do_transpose,
                                              const TilingInfo &tilingInfo,
                                              const LayerInfo &layerInfo) {
  int stride = tilingInfo.stride[0];
  int OHP = tilingInfo.oh;
  int OWP = tilingInfo.ow;
  int OCP = tilingInfo.oc;
  int ICP = tilingInfo.ic;
  int ic = tilingInfo.align_ifm[0];
  int kh = tilingInfo.kernel_size;
  int kw = tilingInfo.kernel_size;
  int w_iter = tilingInfo.width_iter;
  int depth_iter = tilingInfo.depth_iter;
  int h_iter = tilingInfo.height_iter;
  int ch_iter = tilingInfo.channel_iter;
  int sv_oh = tilingInfo.ofmsv_chunk[1];
  int sv_ow = tilingInfo.ofmsv_chunk[2];
  int sv_oc = tilingInfo.ofmsv_chunk[0];
  int sv_ih = tilingInfo.ifmsv_chunk[1];
  int sv_iw = tilingInfo.ifmsv_chunk[2];
  int sv_ic = tilingInfo.ifmsv_chunk[0];
  int icg = (int)(sv_ic / ICP);
  int ocg = (int)(sv_oc / OCP);

  // Stride-specific hyperparameter settings
  if (stride == 2) {
    OHP = 1;
    OWP = 4;
    OCP = 8;
    ICP = 8;
    // std::cout << "\tOHP, OWP, OCP, ICP: ", OHP, OWP, OCP, ICP;
  } else if (stride == 1) {
    OHP = 1;
    OWP = 8;
    OCP = 8;
    ICP = 8;
    // std::cout << "\tOHP, OWP, OCP, ICP: ", OHP, OWP, OCP, ICP;
  } else {
    std::cout << "\tstride error! now just support 1 and 2!";
    throw new std::logic_error("error - given stride not supported");
  }

  // Pre-setting for TDM and coefficient types
  std::string tdm_type = (tilingInfo.ifm_type.find("uint") != std::string::npos)
                             ? "uint32"
                             : "int32";
  std::string coeff_type = "int64";
  int batch = 1;

  // Calculate padded input channels
  int ic_ori = ic;
  ic = (int)(std::ceil(ic_ori / aie_ic_aligned_size) * aie_ic_aligned_size);

  // Get data sizes
  auto type_sizes = get_type_size({tilingInfo.ifm_type, tilingInfo.wgt_type,
                                   tilingInfo.ofm_type, coeff_type});
  auto ifmBytes = type_sizes[0];
  auto ofmBytes = type_sizes[2];

  // ############## init ddflow ###############
  LayerParams layer_params;
  layer_params.set_sv_iw((uint8_t)sv_iw);
  layer_params.set_sv_ih((uint8_t)sv_ih);
  layer_params.set_icg((uint8_t)icg);
  layer_params.set_ocg((uint8_t)ocg);
  layer_params.set_log2stride((uint8_t)std::log2(stride));
  layer_params.set_ofm_depth_iters((uint8_t)ch_iter);
  layer_params.set_ifm_depth_iters((uint8_t)depth_iter);
  layer_params.set_ch_in_depth_split((uint8_t)1);
  layer_params.set_adf_colums((uint8_t)2);
  // ############## init ddflow ###############

  // Geometric parameter initializatiin
  auto Kx_g = kw;
  auto Ky_g = kh;
  auto Ci_g = sv_ic / ICP;
  auto X_g = sv_ow / OCP;
  auto Y_g = sv_oh / OHP;
  auto Co_g = (int)(sv_oc / OCP);

  // Inner and outer product sizes
  auto inner_g = Kx_g * Ky_g * Ci_g;
  auto outer_g = X_g * Y_g * Co_g;

  // Shifting values for parameters.
  auto shift_tdm = layerInfo.shift_conv;
  auto shift_res = layerInfo.shift_out;
  auto shift_bias = 0;

  // Input feature map strides
  auto step_Kx = ICP * ifmBytes;
  auto step_Ky = sv_iw * sv_ic * ifmBytes;
  auto step_Ci = sv_iw * ICP * ifmBytes;
  auto step_Xi = ICP * ifmBytes * stride;
  auto step_Yi = sv_iw * sv_ic * ifmBytes * stride;

  // Output feature map Strides
  auto step_Xo = OCP * ofmBytes;
  auto step_Yo = sv_ow * sv_oc * ofmBytes;
  auto step_Co = OCP * sv_ow * ofmBytes;

  // Wrapper type for the type of operation to be performed
  auto wrapper_type = 0;
  if (do_maxpool) {
    wrapper_type = 5;
  } else if (do_transpose) {
    wrapper_type = 6;
  }

  KernelControl control;
  // Setting kernels based on the input data type
  if (tilingInfo.ifm_type == "int8") {
    control.set_zero_init(true);
    control.set_sign_N(true);
    control.set_sign_O(true);
    control.set_reserved_3(1);
  } else {
    control.set_zero_init(true);
  }
  uint32_t MLKernelControl = control.data();

  // Max pooling parameter initialization
  auto shift_Qb = 0;
  auto ifm_sv_width = 0;
  auto ifm_sv_height = 0;
  auto ifm_sv_depth = 0;
  auto ofm_sv_depth = 0;
  int maxpool_ksize = 0;
  int maxpool_stride = 0;

  // Max pooling kernels

  if (do_maxpool) {
    // Max pooling params initialization
    ifm_sv_width = sv_ow;
    ifm_sv_height = sv_oh;
    ifm_sv_depth = sv_oc;
    maxpool_ksize = (tilingInfo.pool_ksize[1] & 0xF) +
                    ((tilingInfo.pool_ksize[0] << 4) & 0xFF);
    maxpool_stride = (int(log2(tilingInfo.pool_strides[0])) << 4) +
                     int(log2(tilingInfo.pool_strides[1]));
  }
  int manual_bd_pad_val = 0;

  ConvLayerParams conv_layer_params;

  conv_layer_params.set_w_iter((uint8_t)w_iter);
  conv_layer_params.set_h_iter((uint8_t)h_iter);
  conv_layer_params.set_depth_iter((uint8_t)depth_iter);
  conv_layer_params.set_h_iter((uint8_t)h_iter);
  conv_layer_params.set_ch_iter((uint8_t)ch_iter);
  conv_layer_params.set_kx_g((uint8_t)kw);
  conv_layer_params.set_ky_g((uint8_t)kh);
  conv_layer_params.set_ci_g((uint8_t)Ci_g);
  conv_layer_params.set_s_g((uint8_t)stride);
  conv_layer_params.set_n_g((uint8_t)batch);
  conv_layer_params.set_x_g((uint8_t)X_g);
  conv_layer_params.set_y_g((uint8_t)Y_g);
  conv_layer_params.set_co_g((uint8_t)Co_g);

  conv_layer_params.set_inner_g((uint16_t)inner_g);
  conv_layer_params.set_outer_g((uint16_t)outer_g);

  conv_layer_params.set_shift_tdm(shift_tdm);
  conv_layer_params.set_shift_res(shift_res);

  conv_layer_params.set_shift_bias((uint16_t)shift_bias);
  conv_layer_params.set_step_kx((uint16_t)step_Kx);
  conv_layer_params.set_step_ky((uint16_t)step_Ky);
  conv_layer_params.set_step_ci((uint16_t)step_Ci);
  conv_layer_params.set_step_xi((uint16_t)step_Xi);
  conv_layer_params.set_step_yi((uint16_t)step_Yi);
  conv_layer_params.set_step_xo((uint16_t)step_Xo);
  conv_layer_params.set_step_yo((uint16_t)step_Yo);
  conv_layer_params.set_step_co((uint16_t)step_Co);

  conv_layer_params.set_wrapper_type((uint32_t)wrapper_type);
  conv_layer_params.set_mlkernel_control((uint32_t)MLKernelControl);
  conv_layer_params.set_c1((uint32_t)layerInfo.c1);
  conv_layer_params.set_c2((uint32_t)layerInfo.c2);

  conv_layer_params.set_shift_qb((uint8_t)shift_Qb);
  conv_layer_params.set_shift_out((uint8_t)layerInfo.shift_out);
  conv_layer_params.set_ifm_sv_width((uint8_t)ifm_sv_width);
  conv_layer_params.set_ifm_sv_height((uint8_t)ifm_sv_height);
  conv_layer_params.set_ifm_sv_depth((uint8_t)ifm_sv_depth);
  conv_layer_params.set_ofm_sv_depth((uint8_t)ofm_sv_depth);
  conv_layer_params.set_maxpool_ksize((uint8_t)maxpool_ksize);
  conv_layer_params.set_maxpool_stride((uint8_t)maxpool_stride);
  conv_layer_params.set_manual_bd_pad_val((uint8_t)manual_bd_pad_val);

  return {layer_params.data(), conv_layer_params.data()};
}

void compare_vectors_and_print(const std::vector<uint8_t> &vec1,
                               const std::vector<uint8_t> &vec2) {
  // Ensure vectors have equal sizes
  if (vec1.size() != vec2.size()) {
    std::cerr << "Error: Vectors must have the same size." << std::endl;
    return;
  }

  const std::string GREEN = "\033[92m";
  const std::string RED = "\033[91m";
  const std::string RESET = "\033[0m";

  for (std::size_t i = 0; i < vec1.size(); ++i) {
    std::string equal_str = (vec1[i] == vec2[i]) ? GREEN + "Equal" + RESET
                                                 : RED + "Not Equal" + RESET;
    std::cout << std::setw(5) << i;
    std::cout << "  " << std::setw(3) << (int)vec1[i] << "       "
              << std::setw(3) << (int)vec2[i] << "                " << equal_str
              << std::endl;
  }
}

std::string
generate_tiling_key(const std::vector<int32_t> &input_shape,
                    const std::vector<int32_t> &output_shape,
                    const std::vector<int32_t> &weight_shape,
                    const std::vector<int32_t> &maxpool_kernel_shape,
                    const std::vector<int32_t> &maxpool_stride) {
  std::string tiling_key;
  for (auto dim : input_shape) {
    tiling_key += std::to_string(dim) + "_";
  }
  for (auto dim : output_shape) {
    tiling_key += std::to_string(dim) + "_";
  }
  for (auto dim : weight_shape) {
    tiling_key += std::to_string(dim) + "_";
  }
  for (auto dim : maxpool_kernel_shape) {
    tiling_key += std::to_string(dim) + "_";
  }
  for (auto dim : maxpool_stride) {
    tiling_key += std::to_string(dim) + "_";
  }
  /* ignore last "_" */
  tiling_key = tiling_key.substr(0, tiling_key.size() - 1);
  // std::cout << "Tiling filename : " << tiling_key << std::endl;
  return tiling_key;
}

std::vector<std::vector<uint8_t>>
computeLayerParams(const TilingInfo &tilingInfo, const LayerInfo &layerInfo) {
  /* mswbjvw specific */
  auto ofm_shape = tilingInfo.align_ofm_shape;

  auto do_maxpool = false;
  auto do_transpose = false;
  if (!tilingInfo.pool_ksize.empty()) {
    do_maxpool = true;
    do_transpose = true;
  }

  return computeUtil(do_maxpool, do_transpose, tilingInfo, layerInfo);
}
} // namespace conv_lp
