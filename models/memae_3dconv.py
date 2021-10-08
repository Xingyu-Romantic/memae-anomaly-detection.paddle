from __future__ import absolute_import, print_function
import paddle
from paddle import nn

from models import MemModule

class AutoEncoderCov3DMem(nn.Layer):
    def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
        super(AutoEncoderCov3DMem, self).__init__()
        print('AutoEncoderCov3DMem')
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3D(self.chnum_in, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3D(feature_num_2),
            nn.LeakyReLU(0.2),
            nn.Conv3D(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3D(feature_num),
            nn.LeakyReLU(0.2),
            nn.Conv3D(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3D(feature_num_x2),
            nn.LeakyReLU(0.2),
            nn.Conv3D(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3D(feature_num_x2),
            nn.LeakyReLU(0.2)
        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=feature_num_x2, shrink_thres =shrink_thres)
        self.decoder = nn.Sequential(
            nn.Conv3DTranspose(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3D(feature_num_x2),
            nn.LeakyReLU(0.2),
            nn.Conv3DTranspose(feature_num_x2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3D(feature_num),
            nn.LeakyReLU(0.2),
            nn.Conv3DTranspose(feature_num, feature_num_2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3D(feature_num_2),
            nn.LeakyReLU(0.2),
            nn.Conv3DTranspose(feature_num_2, self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1))
        )

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att}