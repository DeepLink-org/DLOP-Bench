import torch
from long_tail_bench.core.executer import Executer


def get_dense_locations(locations, stride, dense_points, device, tocaffe):
    if dense_points <= 1:
        return locations
    center = 0
    step = stride // 4
    l_t = [center - step, center - step]
    r_t = [center + step, center - step]
    l_b = [center - step, center + step]
    r_b = [center + step, center + step]
    if dense_points == 4:
        if not tocaffe:
            points = torch.cuda.FloatTensor([l_t, r_t, l_b, r_b],
                                            device=device)
        else:
            points = torch.FloatTensor([l_t, r_t, l_b, r_b], device=device)
    elif dense_points == 5:
        if not tocaffe:
            points = torch.cuda.FloatTensor(
                [l_t, r_t, [center, center], l_b, r_b], device=device)
        else:
            points = torch.FloatTensor([l_t, r_t, [center, center], l_b, r_b],
                                       device=device)

    else:
        print("dense points only support 1, 4, 5")
    points.reshape(1, -1, 2)
    locations = locations.reshape(-1, 1, 2).to(points)
    dense_locations = points + locations
    dense_locations = dense_locations.view(-1, 2)
    return dense_locations


def compute_locations_per_lever(h,
                                w,
                                stride,
                                device="cuda",
                                dense_points=1,
                                tocaffe=False):
    shifts_x = torch.arange(0,
                            w * stride,
                            step=stride,
                            dtype=torch.float32,
                            device=device)
    shifts_y = torch.arange(0,
                            h * stride,
                            step=stride,
                            dtype=torch.float32,
                            device=device)
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    locations = get_dense_locations(locations, stride, dense_points, device,
                                    tocaffe)
    return locations


def args_generator(N):
    w = 4
    stride = 4

    return [N, w, stride]


def executer_creator():
    return Executer(compute_locations_per_lever, args_generator)
