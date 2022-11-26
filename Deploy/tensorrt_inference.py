import tensorrt as trt
from Utils.BaseClass import BaseInference
from collections import OrderedDict,namedtuple
import torch
import numpy as np
from Utils.Tools import letterbox


class YoloTRTInference(BaseInference):
    def __init__(self, trt_path, output_names=["output"], device="cuda:0"):
        super(YoloTRTInference, self).__init__()
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(trt_path, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()
        self.output_names = output_names

    def preprocess(self, images:list):
        new_image = []
        for image in images:
            image = image.copy()
            image, ratio, dwdh = letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            image = image.astype(np.float32)
            new_image.append(image)
        return np.concatenate(new_image, axis=0)

    def forward(self, images):
        self.binding_addrs['images'] = int(images.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        output = [self.bindings[name].data for name in self.output_names]
        return output

    def __call__(self, images):
        images = self.preprocess(images)
        output = self.forward(images)
        return output