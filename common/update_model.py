import onnx
import os
from onnx import optimizer

original_model = onnx.load('model.onnx')

all_passes = optimizer.get_available_passes()
print("Available optimization passes:")
for p in all_passes:
    print('\t{}'.format(p))
print()

passes = ['eliminate_identity', 'eliminate_nop_pad', 'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_conv']
onnx_optimized = 'model.onnx'
optimized_model = optimizer.optimize(original_model, passes)
onnx.save(optimized_model, onnx_optimized)