import torch
import torch.nn.functional as F
from torch import nn

from core.utils.bboxes import xyxy_to_xywh


class MultiBoxLoss:
    def __init__(self, anchors, threshold, variance, negpos_ratio, device):
        self.device = device
        # torch.Tensor, shape: (先验框总数(8732), 4)
        self.default_boxes = torch.from_numpy(xyxy_to_xywh(anchors)).to(device)
        self.default_boxes.requires_grad = False
        self.default_boxes = self.default_boxes.unsqueeze(dim=0)  # shape: (1, 8732, 4)
        self.threshold = threshold
        self.variance = variance
        self.negpos_ratio = negpos_ratio
        self.scale_xy = 1.0 / self.variance[0]  # 10
        self.scale_wh = 1.0 / self.variance[1]  # 5

    def _location_vec(self, loc):
        g_cxcy = self.scale_xy * (loc[..., :2] - self.default_boxes[..., :2]) / self.default_boxes[..., 2:]
        g_wh = self.scale_wh * torch.log(loc[..., 2:] / self.default_boxes[..., 2:])
        return torch.cat(tensors=(g_cxcy, g_wh), dim=-1)

    def __call__(self, y_true, y_pred):
        """
        :param y_true: torch.Tensor, shape: (batch_size, 8732, 5(cx, cy, w, h, class_index))
        :param y_pred: (loc, conf), 其中loc的shape是(batch_size, 8732, 4), conf的shape是(batch_size, 8732, num_classes)
        :return:
        """
        # ploc: (batch, 8732, 4)
        # plabel: (batch, 8732, num_classes)
        ploc, plabel = y_pred
        gloc = y_true[..., :-1]  # (batch_size, 8732, 4)
        glabel = y_true[..., -1].long()  # (batch_size, 8732)

        # 筛选正样本
        mask = glabel > 0  # (batch_size, 8732)
        # 每一张图片的正样本个数
        pos_num = mask.sum(dim=1)  # (batch_size)

        # 偏移量
        vec_gd = self._location_vec(gloc)  # (batch_size, 8732, 4)
        # 位置损失
        loc_loss = F.smooth_l1_loss(ploc, vec_gd, reduction="none").sum(dim=-1)  # (batch_size, 8732)
        # 只计算正样本的位置损失
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # (batch_size)

        # Hard Negative Mining
        con = F.cross_entropy(torch.permute(plabel, dims=(0, 2, 1)), glabel, reduction="none")  # (batch_size, 8732)
        # 获取负样本
        con_neg = con.clone()
        # 将正样本对应位置处的分类损失置0
        con_neg[mask] = torch.tensor(0.0)
        # 排序，得到一个索引，它的值表示这个位置的元素第几大
        _, con_idx = con_neg.sort(1, descending=True)
        _, con_rank = con_idx.sort(1)
        # 负样本的数量是正样本的self.negpos_ratio倍，但不能超过8732
        neg_num = torch.clamp(self.negpos_ratio * pos_num, max=mask.size(1)).unsqueeze(-1)  # (batch_size, 1)
        neg_mask = con_rank < neg_num  # (batch_size, 8732)

        # 总的分类损失包括：正样本的分类损失，选取的负样本的分类损失
        con_loss = (con * (mask.float() + neg_mask.float())).sum(1)  # (batch_size)

        total_loss = loc_loss + con_loss
        # 避免出现图片中没有正样本的情况
        num_mask = (pos_num > 0).float()
        # 防止分母为0
        pos_num = pos_num.float().clamp(min=1e-6)
        total_loss = (total_loss * num_mask / pos_num).mean(dim=0)
        loss_l = (loc_loss * num_mask / pos_num).mean(dim=0)
        loss_c = (con_loss * num_mask / pos_num).mean(dim=0)
        return total_loss, loss_l, loss_c


class MultiBoxLossV2:
    def __init__(self, neg_pos_ratio, num_classes):
        # self.device = device
        # torch.Tensor, shape: (先验框总数(8732), 4)
        # self.default_boxes = torch.from_numpy(xyxy_to_xywh(anchors)).to(device)
        # self.default_boxes.requires_grad = False
        # self.default_boxes = self.default_boxes.unsqueeze(dim=0)  # shape: (1, 8732, 4)
        # self.threshold = threshold
        # self.variance = variance
        self.neg_pos_ratio = neg_pos_ratio
        # self.scale_xy = 1.0 / self.variance[0]  # 10
        # self.scale_wh = 1.0 / self.variance[1]  # 5
        self.negatives_for_hard = torch.FloatTensor([100])[0]
        self.num_classes = num_classes + 1
        self.background_label_id = 0
        self.alpha = 0.5

    @staticmethod
    def _l1_smooth_loss(y_true, y_pred):
        abs_loss = torch.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = torch.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
        return torch.sum(l1_loss, -1)

    @staticmethod
    def _softmax_loss(y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-7)
        softmax_loss = -torch.sum(y_true * torch.log(y_pred), dim=-1)
        return softmax_loss

    def __call__(self, y_true, y_pred):
        """
        :param y_true: torch.Tensor, shape: (batch_size, 8732, 4(cx, cy, w, h) + num_classes(one_hot) + 1)
        :param y_pred: (loc, conf), 其中loc的shape是(batch_size, 8732, 4), conf的shape是(batch_size, 8732, num_classes)
        :return:
        """
        num_boxes = y_true.size(1)
        y_pred = torch.cat([y_pred[0], nn.Softmax(-1)(y_pred[1])], dim=-1)

        # --------------------------------------------- #
        #   分类的loss
        #   batch_size,8732,21 -> batch_size,8732
        # --------------------------------------------- #
        conf_loss = self._softmax_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:])

        # --------------------------------------------- #
        #   框的位置的loss
        #   batch_size,8732,4 -> batch_size,8732
        # --------------------------------------------- #
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # --------------------------------------------- #
        #   获取所有的正标签的loss
        # --------------------------------------------- #
        pos_loc_loss = torch.sum(loc_loss * y_true[:, :, -1],
                                 dim=1)
        pos_conf_loss = torch.sum(conf_loss * y_true[:, :, -1],
                                  dim=1)

        # --------------------------------------------- #
        #   每一张图的正样本的个数
        #   num_pos     [batch_size,]
        # --------------------------------------------- #
        num_pos = torch.sum(y_true[:, :, -1], dim=-1)

        # --------------------------------------------- #
        #   每一张图的负样本的个数
        #   num_neg     [batch_size,]
        # --------------------------------------------- #
        num_neg = torch.min(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        # 找到了哪些值是大于0的
        pos_num_neg_mask = num_neg > 0
        # --------------------------------------------- #
        #   如果所有的图，正样本的数量均为0
        #   那么则默认选取100个先验框作为负样本
        # --------------------------------------------- #
        has_min = torch.sum(pos_num_neg_mask)
        num_neg_batch = torch.sum(num_neg) if has_min > 0 else self.negatives_for_hard

        # --------------------------------------------- #
        #   对预测结果进行判断，如果该先验框没有包含物体
        #   那么它的不属于背景的预测概率过大的话
        #   就是难分类样本
        # --------------------------------------------- #
        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes - 1

        # --------------------------------------------- #
        #   batch_size,8732
        #   把不是背景的概率求和，求和后的概率越大
        #   代表越难分类。
        # --------------------------------------------- #
        max_confs = torch.sum(y_pred[:, :, confs_start:confs_end], dim=2)

        # --------------------------------------------------- #
        #   只有没有包含物体的先验框才得到保留
        #   我们在整个batch里面选取最难分类的num_neg_batch个
        #   先验框作为负样本。
        # --------------------------------------------------- #
        max_confs = (max_confs * (1 - y_true[:, :, -1])).view([-1])

        _, indices = torch.topk(max_confs, k=int(num_neg_batch.cpu().numpy().tolist()))

        neg_conf_loss = torch.gather(conf_loss.view([-1]), 0, indices)

        # 进行归一化
        num_pos = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))
        # 正样本的分类损失 + 负样本的分类损失
        conf_loss = (torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss)) / torch.sum(num_pos)
        # 正样本的回归损失
        loc_loss = torch.sum(pos_loc_loss) / torch.sum(num_pos)
        # total_loss = torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        # total_loss = total_loss / torch.sum(num_pos)
        total_loss = conf_loss * (1 - self.alpha) + loc_loss * self.alpha
        return total_loss, loc_loss, conf_loss