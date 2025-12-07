
#####
# dummy input to feed some data to onnx models
#####
import numpy as np
data = np.ones((1,3,244,244), dtype=np.float32)
data.tofile("data.bin")
