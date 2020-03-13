import sys
import cntk as C

print(sys.argv[1])
Z = C.Function.load(sys.argv[1])
Z.save("updated.onnx", format=C.ModelFormat.ONNX)