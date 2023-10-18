# Quantization_pytorch_tensorrt
Here is a example of how to use TensorRT python_quantization toolkit to complete PTQ and QAT.


### 0.Requirement
``` bash
# torch >= 1.9.1
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
pip install tensorrt
```

### 1.Quantization Step

#### -Step 1: Train FP32 model
``` bash
python quantization_code/fp32_train.py
```

#### -Step 2: Get PTQ model
``` bash
python quantization_code/ptq.py
```

#### -Step 3: Get QAT model
``` bash
# With pytorch_quantization, PTQ is a must before QAT.
python quantization_code/qat.py
```

#### -Step 4: Convert .pth to .onnx
``` bash
python quantization_code/convert_onnx.py
```

### 2.Inference on TRT Step

#### -Step 1: Convert .onnx to .trt
``` bash
# Need to prepare calibrated data for INT8 model.
python verification_trt/convert_trt.py
```
#### -Step 2: Inference .trt
``` bash
python verification_trt/infer.py
```


-----------
### Reference
* https://github.com/NVIDIA/TensorRT/tree/release/8.6/tools/pytorch-quantization
* https://blog.csdn.net/qq_43522163/article/details/128735345?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-7-128735345-blog-119053735.235^v38^pc_relevant_default_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-7-128735345-blog-119053735.235^v38^pc_relevant_default_base&utm_relevant_index=12
