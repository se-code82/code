import torch
from torch import nn

import config
from model.encoder import build_encoder
from model.decoder import build_decoder
from model.aspp import build_aspp
from torchsummary import summary
from model.configs.swin_224_7_2level import EncoderConfig
from model.configs.swin_224_7_2level import ASPPConfig
from model.configs.swin_224_7_2level import DecoderConfig
class SwinDeepLab(nn.Module):
    def __init__(self, encoder_config, aspp_config, decoder_config):
        super().__init__()

        self.encoder = build_encoder(encoder_config)

        self.aspp = build_aspp(input_size=self.encoder.high_level_size,
                               input_dim=self.encoder.high_level_dim,
                               out_dim=self.encoder.low_level_dim, config=aspp_config)

        self.decoder = build_decoder(input_size=self.encoder.high_level_size,
                                     input_dim=self.encoder.low_level_dim,
                                     config=decoder_config)

    def run_encoder(self, x):
        low_level, high_level = self.encoder(x)
        return low_level, high_level
    
    def run_aspp(self, x):
        return self.aspp(x)

    def run_decoder(self, low_level, pyramid):
        return self.decoder(low_level, pyramid)

    def run_upsample(self, x):
        return self.upsample(x)

    def forward(self, x):
        low_level, high_level = self.run_encoder(x)
        x = self.run_aspp(high_level)
        x = self.run_decoder(low_level, x)
        
        return x


if __name__ == "__main__":
    model = SwinDeepLab(EncoderConfig, ASPPConfig, DecoderConfig).cuda()
    print(model,(3,224,224))
    summary(model)