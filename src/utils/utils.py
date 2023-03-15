import onnx
import pycuda.driver as cuda
import tensorrt as trt
import torch
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TensorrtBase(object):
    def __init__(self,
                 model,
                 speed_up_weights_root="",
                 onnx_filename="",
                 trt_filename="",
                 gpu_id="0",
                 longer_size=1280,
                 rebuild_engine=False):

        # self.if_dynamic = if_dynamic
        # self.input_shape = input_shape  # (h, w)
        self.speed_up_weights_root = speed_up_weights_root
        os.makedirs(self.speed_up_weights_root, exist_ok=True)
        self.onnx_filename = onnx_filename
        self.trt_filename = trt_filename
        self.gpu_id = gpu_id
        self.device = torch.device('cuda:{}'.format(self.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.longer_size = longer_size

        # get trt engine
        self.get_onnx(model)
        self.trt_engine = self.build_engine(rebuild_engine=rebuild_engine)

        self.context = self.trt_engine.create_execution_context()
        self.buffers = self.allocate_buffer()

    def get_onnx(self, model):
        onnx_file = os.path.join(self.speed_up_weights_root, self.onnx_filename)
        input_names = ["input"]
        output_names = ["output0", "output1", "output2"]
        input_shape = (1, 3, self.longer_size, self.longer_size)
        input = torch.randn(input_shape).to(self.device)
        torch.onnx.export(model,
                          input,
                          onnx_file,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=11,
                          verbose=True)

        onnx.checker.check_model(onnx_file)
        print("==> Passed")

    def build_engine(self,
                     use_fp16=True,
                     batch_size=1,
                     rebuild_engine=False):
        """Build TensorRT Engine
        :use_fp16: set mixed flop computation if the platform has fp16.
        """
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            config = builder.create_builder_config()
            config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)
            # Default workspace is 2G
            config.max_workspace_size = 2 << 30
            if builder.platform_has_fast_fp16 and use_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            # parse ONNX
            onnx_file = os.path.join(self.speed_up_weights_root, self.onnx_filename)
            with open(onnx_file, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            print("===> Completed parsing ONNX file")

            # default = 1 for fixed batch size
            builder.max_batch_size = batch_size
            input_shape = (batch_size, 3, self.longer_size, self.longer_size)
            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, input_shape, input_shape, input_shape)
            config.add_optimization_profile(profile)

            # Remove existing engine file
            trt_file = os.path.join(self.speed_up_weights_root, self.trt_filename)
            if os.path.isfile(trt_file) and not rebuild_engine:
                # If a serialized engine exists, use it instead of building an engine.
                print("Reading engine from file {}".format(trt_file))
                with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                    engine = runtime.deserialize_cuda_engine(f.read())
                    return engine
            else:
                try:
                    os.remove(trt_file)
                except Exception:
                    print(f"Cannot remove existing file: {trt_file}")

                print("===> Creating Tensorrt Engine...")
                engine = builder.build_engine(network, config)
                if engine:
                    with open(trt_file, "wb") as f:
                        f.write(engine.serialize())
                    print("===> Serialized Engine Saved at: ", trt_file)
                else:
                    print("===> build engine error")
                return engine

    def allocate_buffer(self):
        """Allocate buffer
        :dynamic_factor: normally expand the buffer size for dynamic shape
        """
        inputs = []
        outputs = []
        bindings = [None for binding in self.trt_engine]
        stream = cuda.Stream()
        for binding in self.trt_engine:
            binding_idx = self.trt_engine[binding]
            if binding_idx == -1:
                print("Error Binding Names!")
                continue
            # trt.volume() return negtive volue if -1 in shape
            size = abs(trt.volume(self.trt_engine.get_binding_shape(binding))) * self.trt_engine.max_batch_size
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings[binding_idx] = int(device_mem)
            # Append to the appropriate list.
            if self.trt_engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self, inf_in_list):
        """Main function for inference
        :inf_in_list: input list.
        :binding_shape_map: {<binding_name>: <shape>}, leave it to None for fixed shape
        """
        inputs, outputs, bindings, stream = self.buffers
        # transfer input data to device
        for i in range(len(inputs)):
            inputs[i].host = inf_in_list[i]
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        # do inference
        # context.profiler = trt.Profiler()
        self.context.execute_async_v2(bindings=bindings,
                                      stream_handle=stream.handle)
        # copy data from device to host
        for i in range(len(outputs)):
            cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)

        stream.synchronize()
        #out_shape = [[1, 67200, 4], [1, 67200, 10], [1, 67200, 2]]
        loc = outputs[0].host.copy().reshape(1, -1, 4)
        conf = outputs[2].host.copy().reshape(1, -1, 2)
        landms = outputs[1].host.copy().reshape(1, -1, 10)
        return loc, conf, landms

    # def __del__(self):
    #     self.cuda_ctx.pop()
    #     del self.cuda_ctx
