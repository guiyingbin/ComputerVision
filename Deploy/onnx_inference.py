from Utils.BaseClass import BaseInference
import onnxruntime as ort
import numpy as np


class YoloInference(BaseInference):
    def __init__(self, onnx_path, input_names, output_names, cuda=False):
        super(YoloInference, self).__init__()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = input_names
        self.output_names = output_names

    def preprocess(self, images):
        """

        :param images: (B, H, W, C)
        :return:
        """
        images = images.transpose((0, 3, 1, 2))
        images = np.ascontiguousarray(images)
        images = images/255
        return images

    def forward(self, images):
        """
        :param images: numpy.ndarray (B, C, H, W)
        :return:
        """
        output = self.session.run(output_names=self.output_names,
                                  input_feed={self.input_names[0]:images})
        return output

    def __call__(self, images):
        images = self.preprocess(images)
        output = self.forward(images)
        return output
