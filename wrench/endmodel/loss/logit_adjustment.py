import random
from typing import Optional, List
import torch
import torch.nn.functional as F

'''
Pytorch implementation of Logit Adjustment
   Reference: "Long-Tail Learning via Logit Adjustment" 
   Authors: Aditya Krishna Menon and
            Sadeep Jayasumana and
            Ankit Singh Rawat and
            Himanshu Jain and
            Andreas Veit and
            Sanjiv Kumar
   https://arxiv.org/pdf/2007.07314.pdf, ICLR'2021.
'''


def logit_adjustment(
        outputs: torch.Tensor,
        prior: List[float] = None,
        tau: Optional[float] = 1.0,
        device: Optional = None,
):
    log_prior = torch.log(torch.tensor(prior, dtype=torch.float) + 1e-8).to(torch.device(device))
    log_prior = log_prior.expand(outputs.size()[0], -1)
    outputs = outputs + log_prior * tau
    return outputs
