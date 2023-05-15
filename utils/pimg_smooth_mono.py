import numpy as np
import cv2

class PBayer:
    """
    Polarimetric bayer image. Each pixel has a Stokes vector.
    """
    def __init__(self, raw, demosaic_method=cv2.COLOR_BAYER_BG2BGR_EA):
        assert raw.dtype == np.float32
        self.raw = raw.copy()
        self.mask = (raw>=0)
        self.mask = self.mask[::2,::2] * self.mask[::2,1::2] * self.mask[1::2,::2] * self.mask[1::2,1::2] # valid if all 2x2 pixels are valid

        raw_ = np.zeros((self.raw.shape[0]+2, self.raw.shape[1]+2), np.float32)
        raw_[2:, 2:] = self.raw
        raw_[:2, 2:] = self.raw[:2, :]
        raw_[2:, :2] = self.raw[:, :2]
        raw_[:2, :2] = self.raw[:2, :2]
        
        p090 = raw_[2::2, 2::2] # |
        p045 = raw_[2::2, 1:-2:2]/2 + raw_[2::2, 3::2]/2
        p135 = raw_[1:-2:2, 2::2]/2 + raw_[3::2, 2::2]/2
        p000 = raw_[1:-2:2, 1:-2:2]/4 + raw_[3::2, 1:-2:2]/4 + raw_[1:-2:2, 3::2]/4 + raw_[3::2, 3::2]/4

        s0 = (p000 + p090 + p045 + p135)/2
        s1 = p000 - p090
        s2 = p045 - p135

        self.stokes = np.zeros((*s0.shape, 4))
        self.stokes[...,0] = s0
        self.stokes[...,1] = s1
        self.stokes[...,2] = s2

        # Stokes vector H x W x 3
        self.svec = np.dstack([s0, s1, s2])
        # Filter angles 4
        self.filt = np.array([0, np.pi/4, np.pi/2, np.pi/4*3])
        # Polarized images H x W x 4
        self.fimg = np.dstack([p000, p045, p090, p135])

        assert self.svec.shape == (self.raw.shape[0]//2, self.raw.shape[1]//2, 3)
        assert self.fimg.shape == (self.raw.shape[0]//2, self.raw.shape[1]//2, 4)

        # H x W
        self.I_dc = s0 / 2
        self.DoLP = np.clip(np.sqrt(s1**2 + s2**2) / (s0 + 1e-9), 0, 1)
        self.AoLP = np.arctan2(s2, s1)
        self.AoLP /= 2

        # H x W
        self.update_conf()

        self.I_max = self.I_dc + self.DoLP * self.I_dc
        self.I_min = self.I_dc - self.DoLP * self.I_dc

        self.BGR_dc = cv2.cvtColor((self.I_dc * 65535).astype(np.uint16), demosaic_method) / 65535.0  # fixme
        assert self.raw.shape[0] == self.BGR_dc.shape[0] * 2
        assert self.raw.shape[1] == self.BGR_dc.shape[1] * 2

    def update_conf(self):
        self.conf = []
        for i in range(4):
            self.conf.append( self.I_dc + self.DoLP * self.I_dc * np.cos(2*self.AoLP - 2 * self.filt[i]) - self.fimg[:,:,i] )
        self.conf = np.linalg.norm(np.dstack(self.conf), axis=2)
        assert self.conf.shape == (self.fimg.shape[0], self.fimg.shape[1]), self.conf.shape
