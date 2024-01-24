from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class NKDLoss(nn.Module):

    """ PyTorch version of NKD (https://arxiv.org/abs/2303.13005)
    
    The problem is that KD calculates loss on the target and non-target classes, 
    whereas student CE only focuses on the target class.
    
    *Normalized KD* 
    DECOUPLES the KD loss into a combination of the target loss 
    (like the original student CE loss) and the non-target
    loss in CE form.
    
    Gist: Better use of soft labels of teacher :) 
    
    Suggested values for gamma and temp are 1.5 and 1.0, respectively.
    
    NKD always applies temperature λ > 1 to make the logit become
    smoother, which causes the logit contains more non-target
    distribution knowledge
    
    """

    def __init__(self,
                 name,
                 use_this,
                 temperature=1.0,
                 gamma=1.5,
                 ):
        super(NKDLoss, self).__init__()

        self.temperature = temperature #controls the smoothness of teacher soft labels
        self.gamma = gamma #tradeoff for target and non-target loss

    def forward(self, logit_s, logit_t, gt_label): #y_s, y_t 
        
        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        s_i = nn.functional.softmax(logit_s, dim=1) #student softmax
        t_i = nn.functional.softmax(logit_t, dim=1) #teacher softmax
        # N*1
        s_t = torch.gather(s_i, 1, label) #student target confidence
        t_t = torch.gather(t_i, 1, label).detach() #teacher target confidence

        # non target logits using mask
        mask = torch.zeros_like(logit_s).scatter_(1, label, 1).bool()   
        logit_s = logit_s - 1000 * mask
        logit_t = logit_t - 1000 * mask
        
        # N*class
        T_i = nn.functional.softmax(logit_t/self.temperature, dim=1) #teacher non-target scaled softmax -> fixed
        S_i = nn.functional.softmax(logit_s/self.temperature, dim=1) #student non-target scaled softmax -> trainable
        # N*1
        T_t = torch.gather(T_i, 1, label) 
        S_t = torch.gather(S_i, 1, label)
        
        # N*class 
        ## trick to nornalize the non-target confidences to 1
        nt_i = T_i/(1-T_t)
        ns_i = S_i/(1-S_t)
        nt_i[T_i==T_t] = 0
        ns_i[T_i==T_t] = 1

        # original student cross-entropy with teacher's soft labels over batch
        loss_t = - (t_t * torch.log(s_t)).mean() 

        # normalized non-target teacher targets KD with normalized non-target student predictions over batch
        loss_non =  (nt_i * torch.log(ns_i)).sum(dim=1).mean()
        loss_non = - self.gamma * (self.temperature**2) * loss_non #NKD part

        return loss_t + loss_non 