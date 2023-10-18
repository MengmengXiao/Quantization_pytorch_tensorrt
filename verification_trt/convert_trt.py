import tensorrt as trt
import os
from calibrator import Calibrator

LOGGER = trt.Logger(trt.Logger.VERBOSE)

def buildEngine(onnx_file, engine_file, quantification, batch_size, FP16_mode, INT8_mode,
                img_height, img_wdith, calibration_images, calibration_cache):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    parser.parse_from_file(onnx_file)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16*(1<<20))

    if FP16_mode == True:
        config.set_flag(trt.BuilderFlag.FP16)
    elif INT8_mode == True:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(quantification, batch_size, img_height, img_wdith, calibration_images, calibration_cache)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("EXPORT ENGINE FAILED!")
        exit(0)

    with open(engine_file, "wb") as f:
        f.write(engine)


def main():
    quantification = 1  # quantization times
    batch_size = 1
    img_height = 32
    img_wdith = 32
    calibration_images = "./images/"
    onnx_file = "../output/quant_resnet18.onnx"
    engine_file = "resnet18.trt"
    calibration_cache = "./weights/resnet18_calibration.cache"

    '''
    FP16_mode = False & INT8_mode = False  (FP32)
    FP16_mode = True  & INT8_mode = False  (FP16)
    FP16_mode = False & INT8_mode = True   (INT8)
    '''
    FP16_mode = False
    INT8_mode = True

    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)

    print('Load ONNX file from:%s \nStart export, Please wait a moment...'%(onnx_file))
    buildEngine(onnx_file, engine_file, quantification, batch_size, FP16_mode, INT8_mode,
                img_height, img_wdith, calibration_images, calibration_cache)
    print('Export ENGINE success, Save as: ', engine_file)


if __name__ == '__main__':
    main()

