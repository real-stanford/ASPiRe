import torch
import torch.nn as nn
import torch.nn.functional as F
from ASPiRe.modules.network import create_mlp


class preprocessor(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim


class image_preprocessor(preprocessor):

    def __init__(self, input_channel, input_dim, output_dim, fc_net_arch, vision_core, **vision_core_args):
        super().__init__(output_dim)
        if output_dim != 0:
            self.identity = True if len(fc_net_arch) == 0 else False
            self.image_encoder = vision_core(input_channel)
            with torch.no_grad():
                output = self.image_encoder(torch.zeros(1, input_channel, input_dim, input_dim))
                output_shape = torch.flatten(output, start_dim=1).shape
                image_encoder_out_dim = output_shape[1]
                print("vision_core out dim", image_encoder_out_dim)  #(batch_size,64,encoder_out,encoder_out)
            if not self.identity:
                fc = create_mlp(input_dim=image_encoder_out_dim, output_dim=output_dim, net_arch=fc_net_arch)
                self.fc = nn.Sequential(*fc)

    def forward(self, image_input):
        encoder_out = self.image_encoder(image_input)
        encoder_out = torch.flatten(encoder_out, start_dim=1)
        if not self.identity:
            feature = self.fc(encoder_out)
        else:
            feature = encoder_out

        return feature


class vector_preprocessor(preprocessor):

    def __init__(self, input_dim, fc_net_arch, output_dim):
        super().__init__(output_dim)
        self.identity = True if len(fc_net_arch) == 0 else False
        if not self.identity:
            fc = create_mlp(input_dim=input_dim, output_dim=output_dim, net_arch=fc_net_arch)
            self.fc = nn.Sequential(*fc)
        else:
            self.output_dim = input_dim

    def forward(self, input):
        if self.identity:
            feature = input
        else:
            feature = self.fc(input)

        return feature


class dict_preprocessor(preprocessor):

    def __init__(self, input_dim, entry):
        output_dim = input_dim
        super().__init__(output_dim)
        self.entry = entry

    def forward(self, input):
        return input[self.entry]


class mix_preprocessor(preprocessor):

    def __init__(self,
                 output_dim: int,
                 image_preprocessor: image_preprocessor,
                 vector_preprocessor: vector_preprocessor,
                 fc_net_arch: list,
                 input_dim: int = -1):
        # only specify input dim when fc_net_arch is empty
        super().__init__(output_dim)
        self.identity = True if len(fc_net_arch) == 0 else False
        if not self.identity:
            self.image_preprocessor = image_preprocessor
            self.vector_preprocessor = vector_preprocessor
            mix_input_dim = self.image_preprocessor.output_dim + self.vector_preprocessor.output_dim
            # fc = create_mlp(input_dim=mix_input_dim, output_dim=output_dim, net_arch=fc_net_arch)
            # self.fc = nn.Sequential(*fc)
            self.output_dim = mix_input_dim
        else:
            self.output_dim = input_dim

    def forward(self, observations):
        if self.identity:
            #TODO: HARD CODE for now
            feature = observations['vector']
        else:
            #TODO: HARD CODE for now
            image_input, vector_input = observations['image'], observations['vector']
            mix_input = []
            if self.image_preprocessor.output_dim != 0:
                image_out = self.image_preprocessor(image_input)
                mix_input.append(image_out)
            if self.vector_preprocessor.output_dim != 0:
                vector_out = self.vector_preprocessor(vector_input)
                mix_input.append(vector_out)

            mix_input = torch.cat(mix_input, dim=1)
            # feature = self.fc(mix_input)

        return mix_input
