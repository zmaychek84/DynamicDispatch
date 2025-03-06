#pragma once

#include <ops/op_interface.hpp>
#include <ops/ops_common.hpp>
#include <regex>
#include <utils/dpu_mdata.hpp>
// #include <string>
#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

namespace ryzenai {

/*
 * Unary is a class to offload matrix but it could be 1D, 2D, 3D
 * ... KD Unary to AIE. this class uses lite runtime stack to
 * interface with XRT
 */

template <typename InT, typename OutT> class unary : public OpInterface {

protected:
  // You have one eye and you need a name, one eye
  std::string name = "unary";

  // Strix is an all wheel drive
  std::string arch = "4x4";
  // these are operands types and shapes (yep shapes)
  std::map<std::string, std::string> txnbin_operand_header = {
      {"bfloat16", "bfloat16"}};

  std::map<std::string, std::vector<std::tuple<int, int>>>
      default_shapes_; // constructor must do something

  /* actual M x K of matrix A */
  int64_t operand_shape_[2];

  std::once_flag instr_reg_flag_;

  /* XRT BO for input and output */
  xrt::bo a_bo_;
  xrt::bo c_bo_;

  /* size for activation dtype*/
  int operand_dtype_size_;

  /* variables to store profile data */
  double a_copy_time_;
  double a_sync_time_;
  double c_copy_time_;
  double c_sync_time_;
  double run_aie_time_;
  int64_t num_run_aie_;
  uint64_t num_execute_ = 0;

  std::once_flag logger_flag_;
  /* debug flag */
  bool debug_ = false;

  /*xclbin and mc_code selection variables*/
  std::string operand_dtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;
  std::string xclbinname = "xclbin/stx/gelue_4x4_abfloat16cbfloat.xclbin";

  void setup_instr_init(){};

  /*
   * Utility function that setups the instruction registry with transaction
   * binaries.
   */
  void setup_instr_registry();
  /*
   * Utility function that checks if an operands shape is supported before
   * execution.
   */
  bool isSupportedShape(const Tensor &operand) {
    const auto &supported_shapes =
        default_shapes_.find(txn_fname_prefix_)->second;
    for (const auto &supported : supported_shapes) {
      if (std::get<0>(supported) == operand.shape.at(0) &&
          std::get<1>(supported) == operand.shape.at(1)) {
        return true;
      }
    }
    return false;
  }

  // Compute pairwise product
  int p_product(const std::tuple<int, int> &t) {
    return std::get<0>(t) * std::get<1>(t);
  };

  // Find maximum pairwise product of all supported shapes
  int max_pairwise_product(
      const std::vector<std::tuple<int, int>> &supportedMatrixShapes) {
    std::tuple<int, int> max = supportedMatrixShapes[0];
    for (size_t i = 1; i < supportedMatrixShapes.size(); i++) {
      std::tuple<int, int> mat = supportedMatrixShapes[i];
      if (p_product(mat) > p_product(max)) {
        max = mat;
      }
    }
    return p_product(max);
  };

  /*********************** PUBLIC *************************************/
public:
  // this is used in the tests and usually sigmoid error is about 2%
  // https://confluence.amd.com/pages/viewpage.action?spaceKey=XDCG&title=Validation+of+SiLU+and+Elementwise+Mul+on+STX#ValidationofSiLUandElementwiseMulonSTX-Step-2:Integratetiling.hfromMicrokern
  inline static float EPSILON = 0.0861652f; // this is the maximum error for the
                                            // computation of the sigmoid.

  std::string get_name() const { return name; };
  void set_name(std::string n) { name = n; };

  std::string get_instr_key(std::string prefix, size_t m, size_t k) const {
    return get_name() + "_" + prefix + "_" + std::to_string(m) + "_" +
           std::to_string(k);
  };

  std::string getXCLBinName() {
    return OpInterface::get_dd_base_dir() + xclbinname;
  };

  void initialize_const_params(
      ConstBufferIO &io, const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override{};
  void initialize_const_params(
      const std::vector<Tensor> &const_params,
      const std::map<std::string, std::any> &attr = {}) override{};

  /* Default 1xK */
  static std::tuple<size_t, size_t>
  extract_MK(const std::vector<Tensor> &inputs) {
    size_t M = 1;
    size_t K = 0;
    if (inputs.at(0).shape.size() == 1) {
      K = inputs.at(0).shape.at(0);
    }
    if (inputs.at(0).shape.size() == 2) {
      M = inputs.at(0).shape.at(0);
      K = inputs.at(0).shape.at(1);
    } else if (inputs.at(0).shape.size() == 3) {
      size_t T = inputs.at(0).shape.at(0);
      M = inputs.at(0).shape.at(1) * T;
      K = inputs.at(0).shape.at(2);
    }
    return std::make_tuple(M, K);
  };

  /* Default 1xK */
  static std::tuple<int, int> extract_MK_(const std::vector<int> &inputs) {
    int M = 1;
    int K = 0;
    if (inputs.size() == 1) {
      K = inputs.at(0);
    }
    if (inputs.size() == 2) {
      M = inputs.at(0);
      K = inputs.at(1);
    } else if (inputs.size() == 3) {
      int T = inputs.at(0);
      M = inputs.at(1) * T;
      K = inputs.at(2);
    }
    return std::make_tuple(M, K);
  };

  void build(const std::string &operand_dtype, bool load_xrt);

  unary(const std::string &operand_dtype, bool load_xrt) {
    build(operand_dtype, load_xrt);
  };
  unary(const std::string &name, const std::string &operand_dtype,
        bool load_xrt) {
    set_name(name);
    build(operand_dtype, load_xrt);
  };
  void execute(std::vector<Tensor> &input,
               std::vector<Tensor> &output) override;
  void execute(std::vector<xrt::bo> &input,
               std::vector<xrt::bo> &output) override {

    // prepare inst_bo and param_bo
    auto instr_bo_key = get_instr_key(txn_fname_prefix_, (int)operand_shape_[0],
                                      (int)operand_shape_[1]);
    xrt::bo instr_bo = instr_reg_.get_instr_bo(instr_bo_key);

    size_t instr_bo_words = instr_bo.size() / sizeof(int);
    auto kernel_ = xrt_ctx_->get_kernel();
    // launch the kernel
    xrt::run run;
    run = kernel_(2, instr_bo, instr_bo_words,
                  input[0].address() + DDR_AIE_ADDR_OFFSET,
                  output[0].address() + DDR_AIE_ADDR_OFFSET, 0, 0, 0);
    run.wait2();
  }
  void set_kernel_shape(std::vector<size_t> shape) {
    operand_shape_[0] = shape.at(0);
    operand_shape_[1] = shape.at(1);
  }

  void debug(bool enable) { debug_ = enable; }

  const std::vector<uint8_t> get_transaction_bin(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override;

  const std::vector<uint8_t> get_super_kernel_params(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override {
    return {};
  };

  std::vector<OpArgMap> get_buffer_reqs(
      std::vector<Tensor> &input, std::vector<Tensor> &output,
      const std::map<std::string, std::any> &attr = {}) const override {

    size_t M1 = input.at(0).shape.at(1);
    size_t N1 = input.at(0).shape.at(2);
    size_t M2 = output.at(1).shape.at(1);
    size_t N2 = output.at(1).shape.at(2);

    if ((M1 != M2) || (N1 != N2)) {
      throw std::runtime_error(
          "Dimensions of all tensors should be equal for silu op\n");
    }
    size_t input_1_bo_size = (M1 * N1 * sizeof(InT));
    size_t output_bo_size = (M1 * N1 * sizeof(OutT));

    std::vector<OpArgMap> arg_map{
        {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_1_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 1, 1, 0, output_bo_size},
    };
    return arg_map;
  };

  // this is to make sure you can customize any location of your bins
  void set_xclbinname(std::string a) { xclbinname = a; };
  std::string get_xclbinname() {
    return OpInterface::get_dd_base_dir() + xclbinname;
  };

  void populate_default_shapes(const std::string &operand_dtype) {
    default_shapes_[get_name() + txnbin_operand_header[operand_dtype]] =
        std::vector<std::tuple<int, int>>();
    default_shapes_[get_name() + txnbin_operand_header[operand_dtype]]
        .push_back(std::make_tuple(1, 11008));
  };

  void look_up_bins_2() {
    std::string path = get_xclbinname();
    for (const auto &entry : fs::directory_iterator(path)) {
      std::cout << entry.path() << std::endl;
    }
  };

  void look_up_bins(std::string path, std::string name = "silu",
                    std::string arch = "4x4") {

    std::string fsep = "/";
    std::string attsep = "_";
    std::string ext = ".xclbin";

    std::vector<std::string> bins;

    for (const auto &entry : fs::directory_iterator(path)) {
      bins.push_back(entry.path().string());
    }
    // the naming convention :
    // file_system + / name _ arch _ operands attributes _ sizes _ extras .[bin,
    // xclbin] name = letter+ ( _ name )* arch = N x N operand attributes =
    // [letter + type]+ type = [int, int8, int16, bf16] sizes = number+ (_
    // sizes)* extras = letter

    std::string f_regex = "([.]+/)";
    std::string arch_regex = "(([0-9]+[xX])*[0-9]+_)*";
    std::string name_regex = "(([a-z]+_)+)"; // std::regex::icase);
    std::string op_regex =
        "([abc](int|bfloat|float)*[0-9]+)+_"; // std::regex::icase);
    std::string sh_regex = "(([0-9]+_)+[0-9\\.]*)";
    std::string ex_regex = "(.+bin)";
    std::string end_regex = "*bin";

    // std::cout << f_regex+name_regex+arch_regex << std::endl;
    //                        1+2    3+4         5+6      7 +8      9
    std::regex pattern(name_regex + arch_regex + op_regex + sh_regex +
                           ex_regex, //+end_regex,
                                     // f_regex, //+
                                     //+arch_regex,
                                     //+op_regex+sh_regex+ex_regex,
                       std::regex::icase);

    std::vector<std::vector<int>> Sdescriptions;

    for (const std::string &i : bins) {
      std::string binname, name, arch, op, shapes;
      std::smatch match;
      std::cout << i << std::endl;
      if (std::regex_search(i.begin(), i.end(), match, pattern)) {
        for (std::string m : match) {
          std::cout << "  submatch " << m << '\n';
        }

        binname = match[0];
        if (binname.find(name) < 0) {
          throw std::runtime_error(name +
                                   "IPU Wrapper expect to have transaction "
                                   "bine named after the class.");
        }

        name = match[1];
        arch = match[3];
        if (arch.size() == 0) {
          arch = "4x4_";
        } // By default
        op = match[5];
        if (op.size() == 0) {
          op = "abfloat16cbfloat16_";
        } // By default
        shapes = match[7];
        if (shapes.size() > 0) {
          // std::cout << "S " << shapes << "\n";
          std::vector<int> res = std::vector<int>();
          std::string item, t;
          size_t posl = 0, posr = shapes.find("_", posl);
          item = shapes.substr(posl, posr);
          // std::cout << "I " << item << "\n";
          size_t M = shapes.size();
          while (posl < M) {
            res.push_back(stoi(item));
            posl = posr + 1;
            posr = shapes.find("_", posl);
            posr = (posr < 0) ? M - 1 : posr;
            item = shapes.substr(posl, posr);
            // std::cout << "I " << item << "\n";
          }

          if (default_shapes_.count(name + arch + op) == 0) {
            default_shapes_[name + arch + op] =
                std::vector<std::tuple<int, int>>();
          }
          default_shapes_[name + arch + op].push_back(extract_MK_(res));
        }

        std::cout << "look up" << name << " " << arch << " " << op << " "
                  << shapes << " " << '\n';

        // std::cout << "  [1] " << match[1] << '\n';
        // std::cout << "  [3] " << match[3] << '\n';
      }
    }
  };
  void look_up_bins();
};
} // namespace ryzenai
