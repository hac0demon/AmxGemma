#include "gguf/gguf.h"
#include "gguf/quant.h"
#include <Accelerate/Accelerate.h>
#include <chrono>
#include <iostream>
#include <vector>

int main() {
  ni::gguf::File f("/Users/ivash/models/gemma-4-E4B-it/body.gguf");
  const auto* w = f.find("token_embd.weight");
  std::cout << "token_embd: type=" << ni::gguf::type_name(w->type)
            << " rows=" << w->rows() << " cols=" << w->ne[0] << "\n";

  size_t N = w->rows();
  size_t K = w->ne[0];
  std::vector<float> x(K, 0.1f);
  std::vector<float> y(N, 0.0f);

  auto t0 = std::chrono::high_resolution_clock::now();
  uint32_t type = w->type;
  const uint8_t* base = w->data;
  uint64_t row_bytes = w->row_bytes();
  float* y_dst = y.data();
  const float* x_ptr = x.data();

  dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t j) {
    const uint8_t* row_src = base + j * row_bytes;
    std::vector<float> row_dequant(K);
    ni::gguf::dequantize(type, row_src, row_dequant.data(), K);
    float dot = 0.0f;
    vDSP_dotpr(x_ptr, 1, row_dequant.data(), 1, &dot, K);
    y_dst[j] = dot;
  });
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "dispatch_apply on token_embd took: " << ms << " ms\n";
  return 0;
}
