import torch
import torch.distributed as dist
from long_tail_bench.core.executer import Executer
from collections import OrderedDict


def _parse_losses(losses):
    """Parse the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
            which may be a weighted sum of all losses, log_vars contains
            all the variables to be sent to the logger.
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f"{loss_name} is not a tensor or list of tensors")

    loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

    log_vars["loss"] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))

        # FixMe: JIT can not trace numpy now
        log_vars[loss_name] = loss_value.item()
    return loss, log_vars


def args_adaptor(np_args):
    top1_acc = torch.from_numpy(np_args[0]).cuda()
    top5_acc = torch.from_numpy(np_args[1]).cuda()
    loss_cls = torch.from_numpy(np_args[2]).cuda()
    loss_cls.requires_grad = True
    losses = {"top1_acc": top1_acc, "top5_acc": top5_acc, "loss_cls": loss_cls}
    return [losses]


def executer_creator():
    return Executer(_parse_losses, args_adaptor)
