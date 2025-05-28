import numpy as np

class CNNPadder:
    def __init__(self, pad):
        self.pad = pad
        self.mode = 'reflect'
        self.constant_values = 0
        self.end_values = 0
        self.stat_length = None

    def set_mode(self, mode):
        supported = [
            'reflect', 'symmetric', 'edge', 'constant', 'wrap',
            'linear_ramp', 'maximum', 'minimum', 'median', 'mean'
        ]
        self.mode = mode

    def set_constant(self, val):
        self.constant_values = val

    def set_end_values(self, val):
        self.end_values = val

    def set_stat_length(self, val):
        self.stat_length = val

    def apply(self, img):
        padding_cfg = ((self.pad, self.pad), (self.pad, self.pad), (0, 0))
        kwargs = {}
        if self.mode == 'constant':
            kwargs['constant_values'] = self.constant_values
        if self.mode == 'linear_ramp':
            kwargs['end_values'] = self.end_values
        if self.mode in ['maximum', 'minimum', 'median', 'mean']:
            kwargs['stat_length'] = self.stat_length
        return np.pad(img, padding_cfg, mode=self.mode, **kwargs)
