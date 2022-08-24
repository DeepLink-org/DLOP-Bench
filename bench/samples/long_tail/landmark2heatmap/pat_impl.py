import torch
import torch.nn as nn
from bench.core.executer import Executer


class FaceGenerator_Subnet(nn.Module):
    def __init__(
        self,
        nc_input_x,
        nc_input_y,
        nc_output,
        ngf=64,
        enc_width=[1, 1, 1, 1, 1, 1],
        dec_width=[1, 1, 1, 1, 1],
    ):
        self.ldmk_cnt = nc_input_y
        self.x_bias = torch.linspace(-1.0, 1.0, 256)
        self.y_bias = torch.linspace(-1.0, 1.0, 256)
        self.x_bias = self.x_bias.view(1, 1, 256).cuda()
        self.y_bias = self.y_bias.view(1, 1, 256).cuda()

    def landmark2heatmap(self, landmark, inv_std=20):
        assert landmark.shape[-1] == (
            self.ldmk_cnt << 1), "[ERROR]: shape of landmarks is false"
        x, y = landmark[:, 0::2].reshape(-1, self.ldmk_cnt,
                                         1), landmark[:, 1::2].reshape(
                                             -1, self.ldmk_cnt, 1)

        gaussian_x = torch.exp(
            -torch.sqrt(torch.abs((x - self.x_bias) * inv_std) + 1e-4))
        gaussian_y = torch.exp(
            -torch.sqrt(torch.abs((y - self.y_bias) * inv_std) + 1e-4))

        gaussian_x = torch.unsqueeze(gaussian_x, dim=2)
        gaussian_y = torch.unsqueeze(gaussian_y, dim=3)

        heatmap = torch.matmul(gaussian_y, gaussian_x)

        return heatmap


def args_adaptor(np_args):
    boxes = torch.from_numpy(np_args[0]).cuda()
    return [boxes]


def executer_creator():
    coder_instance = FaceGenerator_Subnet(3, 39, 3)
    return Executer(coder_instance.landmark2heatmap, args_adaptor)
