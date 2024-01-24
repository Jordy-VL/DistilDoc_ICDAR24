from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReg(nn.Module):
    """Convolutional regression for FitNet (feature-map layer)"""

    def __init__(self, s_shape, t_shape):
        super(ConvReg, self).__init__()
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        self.s_H = s_H
        self.t_H = t_H
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        x = self.conv(x)
        if self.s_H == 2 * self.t_H or self.s_H * 2 == self.t_H or self.s_H >= self.t_H:
            return self.relu(self.bn(x)), t
        else:
            return self.relu(self.bn(x)), F.adaptive_avg_pool2d(t, (self.s_H, self.s_H))

    def hidden_states_to_feature_map(self, hidden_states, distill="hint-middle"):
        """
        model.config.num_hidden_layers +1 for patch embeddings
        model.config.patch_size**2 +1 f_s CLS
        model.config.hidden_size
                    #or middle layer
        """
        resnet = True if len(hidden_states) == 5 else False
        if distill == "hint-middle":
            if resnet: #fix for cnns
                middle = 3
                return hidden_states[middle]
            else:
                middle = (len(hidden_states) - 1) // 2
            hidden_states = hidden_states[middle]
        else:
            raise NotImplementedError('distill arg must be "hint-middle"')

        reshaped = hidden_states[:, 1:, :].transpose(1, 2)
        feature_map = reshaped.reshape(
            *reshaped.shape[:2],
            self.s_H,
            self.s_H,
        )  # [bs, hidden_layers,  hidden_size, patch_size, patch_size]

        # assume default: middle (non embedding) layer as hint
        ## the later, the more chance of overregularization
        # middle = patched.shape[1] // 2

        # feature_map = patched[:, middle, :, :]

        # based on distill arghument could still do something with CLS rep
        # cls_rep = hidden_states[:, :, 0, :].transpose(0, 1)  # `[bs, hidden_layers,  hidden_size]`

        return feature_map


class SelfA(nn.Module):
    """Cross-layer Self Attention"""

    def __init__(self, feat_dim, s_n, t_n, soft, factor=4):
        super(SelfA, self).__init__()

        self.soft = soft
        self.s_len = len(s_n)
        self.t_len = len(t_n)
        self.feat_dim = feat_dim

        # query and key mapping
        for i in range(self.s_len):
            setattr(self, "query_" + str(i), MLPEmbed(feat_dim, feat_dim // factor))
        for i in range(self.t_len):
            setattr(self, "key_" + str(i), MLPEmbed(feat_dim, feat_dim // factor))

        for i in range(self.s_len):
            for j in range(self.t_len):
                setattr(self, "regressor" + str(i) + str(j), Proj(s_n[i], t_n[j]))

    def forward(self, feat_s, feat_t):
        sim_s = list(range(self.s_len))
        sim_t = list(range(self.t_len))
        bsz = self.feat_dim

        # similarity matrix
        for i in range(self.s_len):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())
        for i in range(self.t_len):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())

        # calculate student query
        proj_query = self.query_0(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, self.s_len):
            temp_proj_query = getattr(self, "query_" + str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)

        # calculate teacher key
        proj_key = self.key_0(sim_t[0])
        proj_key = proj_key[:, :, None]
        for i in range(1, self.t_len):
            temp_proj_key = getattr(self, "key_" + str(i))(sim_t[i])
            proj_key = torch.cat([proj_key, temp_proj_key[:, :, None]], 2)

        # attention weight: batch_size X No. stu feature X No.tea feature
        energy = torch.bmm(proj_query, proj_key) / self.soft
        attention = F.softmax(energy, dim=-1)

        # feature dimension alignment
        proj_value_stu = []
        value_tea = []
        for i in range(self.s_len):
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(self.t_len):
                s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
                if s_H > t_H:
                    source = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                    target = feat_t[j]
                elif s_H <= t_H:
                    source = feat_s[i]
                    target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))

                proj_value_stu[i].append(getattr(self, "regressor" + str(i) + str(j))(source))
                value_tea[i].append(target)

        return proj_value_stu, value_tea, attention


class Proj(nn.Module):
    """feature dimension alignment by 1x1, 3x3, 1x1 convolutions"""

    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(Proj, self).__init__()
        self.num_mid_channel = 2 * num_target_channels

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear mapping for attention calculation"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)
        self.regressor = nn.Sequential(
            nn.Linear(dim_in, 2 * dim_out),
            self.l2norm,
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim_out, dim_out),
            self.l2norm,
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))

        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class SRRL(nn.Module):
    """ICLR-2021: Knowledge Distillation via Softmax Regression Representation Learning"""

    def __init__(self, *, s_n, t_n):
        super(SRRL, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        setattr(
            self,
            "transfer",
            nn.Sequential(
                conv1x1(s_n, t_n),
                nn.BatchNorm2d(t_n),
                nn.ReLU(inplace=True),
            ),
        )

    def forward(self, feat_s, cls_t):
        feat_s = feat_s.unsqueeze(-1).unsqueeze(-1)
        temp_feat = self.transfer(feat_s)
        trans_feat_s = temp_feat.view(temp_feat.size(0), -1)

        pred_feat_s = cls_t(trans_feat_s)

        return trans_feat_s, pred_feat_s


class ConvProjector(nn.Module):
    def __init__(self, *, s_n, t_n, factor=2):
        super(ConvProjector, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d((1, 1))

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv1d(
                in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups
            )

        # A bottleneck design to reduce extra parameters
        setattr(
            self,
            "transfer",
            nn.Sequential(
                conv1x1(s_n, t_n // factor),
                nn.BatchNorm1d(t_n // factor),
                nn.ReLU(inplace=True),
                conv3x3(t_n // factor, t_n // factor),
                # depthwise convolution
                # conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
                nn.BatchNorm1d(t_n // factor),
                nn.ReLU(inplace=True),
                conv1x1(t_n // factor, t_n),
                nn.BatchNorm1d(t_n),
                nn.ReLU(inplace=True),
            ),
        )

    def forward(self, feat_s, feat_t):
        # Spatial Dimension Alignment
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        if s_H > t_H:
            source = F.adaptive_avg_pool1d(feat_s, (t_H, t_H))
            target = feat_t
        else:
            source = feat_s
            target = F.adaptive_avg_pool1d(feat_t, (s_H, s_H))

        # Channel Alignment
        trans_feat_s = getattr(self, "transfer")(source)
        return trans_feat_s


class SimKD(nn.Module):
    """CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""

    def __init__(self, *, s_n, t_n, patch_dim, factor=2):
        super(SimKD, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups
            )

        # A bottleneck design to reduce extra parameters
        setattr(
            self,
            "transfer",
            nn.Sequential(
                conv1x1(s_n, t_n // factor),
                nn.BatchNorm2d(t_n // factor),
                nn.ReLU(inplace=True),
                conv3x3(t_n // factor, t_n // factor),
                # depthwise convolution
                # conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
                nn.BatchNorm2d(t_n // factor),
                nn.ReLU(inplace=True),
                conv1x1(t_n // factor, t_n),
                nn.BatchNorm2d(t_n),
                nn.ReLU(inplace=True),
            ),
        )

        self.patch_dim = patch_dim

    def hidden_states_to_feature_map(self, hidden_state):
        feature_map = (
            hidden_state[:, 1:, :]
            .transpose(1, 2)
            .reshape(hidden_state.shape[0], hidden_state.shape[-1], self.patch_dim, self.patch_dim)
        )  # [bs, hidden_size, P], remove CLS
        return feature_map

    def forward(self, feat_s, feat_t):
        # Spatial Dimension Alignment
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            target = feat_t
        else:
            source = feat_s
            target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))

        trans_feat_t = target

        # Channel Alignment
        trans_feat_s = getattr(self, "transfer")(source)

        return trans_feat_s, trans_feat_t


class BottleneckMLP(nn.Module):
    """
    simTransKD projector [with or without dropout?]
    """

    def __init__(self, input_dim, output_dim, factor=2):
        super().__init__()
        h1 = h2 = int(input_dim // factor)

        self.fc1 = nn.Linear(input_dim, h1)
        self.bottleneck = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # dropout here?
        x = F.relu(self.bottleneck(x))
        # dropout here?
        x = F.relu(self.fc3(x))
        return x


class SimTransKD(nn.Module):
    """CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""

    "Adapted towards Transformers with MLP-bottleneck projector"
    # TBH does nothing more than the bottleneck MLP above

    def __init__(self, *, s_n, t_n, factor=2):
        super(SimTransKD, self).__init__()
        self.projector = BottleneckMLP(s_n, t_n, factor)

    def forward(self, feat_s):
        # project student features to teacher feature space
        proj_feat_s = self.projector(feat_s)
        return proj_feat_s


class DualSimTransKD(nn.Module):
    """CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""

    "Adapted towards Transformers with MLP-bottleneck projector AND inverter"
    # TBH does nothing more than the bottleneck MLP above

    def __init__(self, *, s_n, t_n, factor=2):
        super(DualSimTransKD, self).__init__()
        self.projector = BottleneckMLP(s_n, t_n, factor)
        self.inverter = BottleneckMLP(t_n, s_n, 1 / factor)  # undo the bottleneck

    def forward(self, feat_s, feat_t):
        # project student features to teacher feature space
        # invert teacher features to student feature space
        proj_feat_s = self.projector(feat_s)
        proj_feat_t = self.inverter(feat_t)
        return proj_feat_s, proj_feat_t


def SimKD_forward(inputs, labels):  # to patch the forward function of the student for deployment
    """
    x = Student.forward_features
    student,projector(x)
    student.classifier(x) #deepcopied from teacher
    """
    raise NotImplementedError
