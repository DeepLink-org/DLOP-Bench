import torch
from torch import nn
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the
            outputs of the model
        2) we supervise each pair of matched ground-truth / prediction
            (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special
                no-object category
            matcher: module able to compute a matching between targets and
                proposals
            weight_dict: dict containing as key the names of the losses and
                as values their relative weight.
            eos_coef: relative classification weight applied to the no-object
                category
            losses: list of all the losses to be applied. See get_loss for
                list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number
            of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only.
            It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets],
                                      device=device)
        # Count the number of predictions that are NOT "no-object" (which is
        #  the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(
            1)  # noqa
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses


def args_generator(num_target_boxes):
    num_classes = 3
    batch_size, num_queries = 4, 100
    pred_logits = torch.randn([batch_size, num_queries, num_classes],
                              device="cuda")
    pred_boxes = torch.randn([batch_size, num_queries, 4], device="cuda")
    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = []
    for _ in range(batch_size):
        labels = torch.randn([num_target_boxes], device="cuda")
        boxes = torch.randn([num_target_boxes, 4], device="cuda")
        target = {"labels": labels, "boxes": boxes}
        targets.append(target)

    indices_ = [
        ([1, 6, 18, 34, 49, 50, 52, 98], [6, 0, 4, 7, 1, 2, 3, 5]),
        ([3, 5], [0, 1]),
        ([33, 54], [1, 0]),
        ([90], [0]),
    ]
    indices = [(
        torch.as_tensor(i, dtype=torch.int64),
        torch.as_tensor(j, dtype=torch.int64),
    ) for i, j in indices_]
    return [outputs, targets, indices, num_target_boxes]


def executer_creator():
    coder_instance = SetCriterion(num_classes=3,
                                  matcher=None,
                                  weight_dict=None,
                                  eos_coef=0.1,
                                  losses=None)
    return Executer(coder_instance.loss_cardinality, args_generator)
