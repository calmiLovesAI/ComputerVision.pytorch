import torch
import torch.nn.functional as F


class FocalLoss:
    def __call__(self, y_true, y_pred):
        """
        :param y_true: 真实值
        :param y_pred: 预测值
        :return:
        """
        pos_idx = torch.eq(y_true, 1).to(torch.float32)
        neg_idx = torch.lt(y_true, 1).to(torch.float32)
        neg_weights = torch.pow(1 - y_true, 4)
        loss = 0
        num_pos = torch.sum(pos_idx)
        pos_loss = torch.log(y_pred) * torch.pow(1 - y_pred, 2) * pos_idx
        pos_loss = torch.sum(pos_loss)
        neg_loss = torch.log(1 - y_pred) * torch.pow(y_pred, 2) * neg_weights * neg_idx
        neg_loss = torch.sum(neg_loss)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class RegL1Loss:
    def __call__(self, y_true, y_pred, mask, index):
        pred = RegL1Loss.gather_feat(y_pred, index)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, y_true * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss

    @staticmethod
    def gather_feat(feat, ind):
        feat = torch.reshape(feat, shape=(feat.size()[0], -1, feat.size()[3]))
        ind = ind.unsqueeze(2).to(torch.int64)
        ind = ind.expand(ind.size()[0], ind.size()[1], feat.size()[2])
        feat = torch.gather(feat, dim=1, index=ind)
        return feat


class CombinedLoss:
    def __init__(self, num_classes, hm_weight, wh_weight, off_weight):
        self.num_classes = num_classes
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight

        self.heatmap_loss_object = FocalLoss()
        self.reg_loss_object = RegL1Loss()
        self.wh_loss_object = RegL1Loss()

    def __call__(self, y_pred, y_true):
        heatmap_true, reg_true, wh_true, reg_mask, indices = y_true
        heatmap = y_pred[..., :self.num_classes]
        reg = y_pred[..., self.num_classes: self.num_classes + 2]
        wh = y_pred[..., -2:]
        heatmap = torch.clamp(input=torch.sigmoid(heatmap), min=1e-4, max=1.0 - 1e-4)
        heatmap_loss = self.heatmap_loss_object(y_true=heatmap_true, y_pred=heatmap)
        off_loss = self.reg_loss_object(y_true=reg_true, y_pred=reg, mask=reg_mask, index=indices)
        wh_loss = self.wh_loss_object(y_true=wh_true, y_pred=wh, mask=reg_mask, index=indices)
        total_loss = self.hm_weight * heatmap_loss + self.off_weight * off_loss + self.wh_weight * wh_loss
        return total_loss
