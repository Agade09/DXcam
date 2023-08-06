import ctypes
import numpy as np
from .base import Processor


class NumpyProcessor(Processor):
    def __init__(self, color_mode):
        self.cvtcolor = None
        self.color_mode = color_mode
        if self.color_mode=='BGRA':
            self.color_mode = None

    def process_cvtcolor(self, image):
        import cv2

        # only one time process
        if self.cvtcolor is None:
            color_mapping = {
                "RGB": cv2.COLOR_BGRA2RGB,
                "RGBA": cv2.COLOR_BGRA2RGBA,
                "BGR": cv2.COLOR_BGRA2BGR,
                "GRAY": cv2.COLOR_BGRA2GRAY
            }
            cv2_code = color_mapping[self.color_mode]
            if cv2_code != cv2.COLOR_BGRA2GRAY:
                self.cvtcolor = lambda image: cv2.cvtColor(image, cv2_code)
            else:
                self.cvtcolor = lambda image: cv2.cvtColor(image, cv2_code)[
                    ..., np.newaxis
                ] 
        return self.cvtcolor(image)

    def process(self, rect, width, height, region, rotation_angle):
        width = region[2] - region[0]
        height = region[3] - region[1]
        if rotation_angle in (90, 270):
            width, height = height, width

        buffer = (ctypes.c_char*height*width*4).from_address(ctypes.addressof(rect.pBits.contents))
        image = np.ndarray((height, width, 4), dtype=np.uint8, buffer=buffer)

        if not self.color_mode is None:
            image = self.process_cvtcolor(image)

        if rotation_angle!=0:
            image = np.rot90(image, k=rotation_angle//90, axes=(1, 0))

        return image
