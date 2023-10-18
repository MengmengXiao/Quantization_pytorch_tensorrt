import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


# Logger
logger = trt.Logger(trt.Logger.WARNING)

def infer(TRT_MODEL_PATH, input_data):
    with open(TRT_MODEL_PATH, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    # Allocate GPU Memory
    output_shape = (10,) 
    output_data = np.empty(output_shape, dtype=np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)
    
    bindings = [int(d_input), int(d_output)]
    
    # Inference
    with engine.create_execution_context() as context:
        cuda.memcpy_htod(d_input, input_data)
        context.execute(1, bindings)
        cuda.memcpy_dtoh(output_data, d_output)
    
    # Output
    predicted_class = np.argmax(output_data)
    print("Output:", output_data)
    print("Predicted class:", predicted_class)

if __name__ == "__main__":
    # TensorRT model
    TRT_MODEL_PATH = "resnet18.trt" 

    # Input: pre-processed feature
    input_data = np.fromfile("input_data.bin", dtype=np.float32).reshape(3,32,32) 

    infer(TRT_MODEL_PATH, input_data)
