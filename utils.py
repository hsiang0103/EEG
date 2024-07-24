import math
import torch
from typing import Union
# helper functions
def processbar(now_process, all, total_len = 30, info = "" , needed_clear = False):
    percent = (now_process / all)
    bar = "■" * int(percent * total_len)
    space = " " * (total_len-len(bar))

    if len(space) > 0 :
        end_char = '\r'
    elif needed_clear:
        # clean whole process bar
        end_char = "\r" + " " * len("|"+bar + space+"|"+info) + "\r"
    else:
        end_char = "\n"

    print("|"+bar + space+"|"+info, end=end_char)

def get_device():
    return 'cuda:1' if torch.cuda.is_available() else 'cpu'

def record_log(record_datas) -> None:
    """
    record data to ./log/

    * Input parameter:
    1. record_datas     : Dictionary (file names = keys)

    * Output:
    None
    """
    for record_data in record_datas:
        with open(f"./log/{record_data}.txt", "w") as f:
            for single_data in record_datas[record_data]:
                f.write(f"{single_data},")
            f.close()

def lrfn(num_epoch, optimizer):
    """
    
    |
    |     ______
    |    /      \
    |   /        \_
    |  /           \___
    | /                \____
    |/                      \_______
    +-------------------------------- epoch
    """
    lr_inital = 1e-5  
    max_lr = 4e-4 
    lr_up_epoch = 10
    lr_sustain_epoch = 5  
    lr_exp = .8  
    if num_epoch < lr_up_epoch:  
        lr = (max_lr - lr_inital) / lr_up_epoch * num_epoch + lr_inital
    elif num_epoch < lr_up_epoch + lr_sustain_epoch:  
        lr = max_lr
    else:  
        lr = (max_lr - lr_inital) * lr_exp ** (num_epoch - lr_up_epoch - lr_sustain_epoch) + lr_inital
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def step_lr(num_epoch, optimizer, sustain_epoch):
    """
        lr
        |
    1e-3|________
        |        |
        |        |
    1e-4|        |________
        |                 |
    1e-5|                 |________
    1e-6|                          |________
        +--------|--------|--------|------------ epoch
                 25       50       75
    """
    lr_inital = 1e-3 
    lr_decay_factor = 1e-1

    lr = lr_inital * (lr_decay_factor ** (num_epoch // sustain_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def warn_up_cosine_lr(num_epoch, warm_up_ratio, optimizer, max_epoch, lr_min, lr_max,start_epoch=0):
    warm_up_epoch = int(warm_up_ratio * (max_epoch - start_epoch)) + start_epoch
    
    if (num_epoch - start_epoch) < (warm_up_epoch - start_epoch):
        lr = lr_max * (num_epoch - start_epoch) / (warm_up_epoch - start_epoch) + lr_min
    else:
        lr = lr_min + (1 / 2) * (lr_max - lr_min) * (1 + math.cos(
            math.pi * (num_epoch - warm_up_epoch) / (max_epoch - warm_up_epoch)
            )
        )

    #warm_up_epoch = int(warm_up_ratio * (max_epoch))
#
    #if num_epoch < (warm_up_epoch):
    #    lr = lr_max * (num_epoch) / (warm_up_epoch) + lr_min
    #else:
    #    lr = lr_min + (1 / 2) * (lr_max - lr_min) * (1 + math.cos(
    #        math.pi * (num_epoch - warm_up_epoch) / (max_epoch - warm_up_epoch)
    #        )
    #    )
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer

class cosine_cycle_anneal_lr():
    def __init__(self, max_epoch, first_period, period_factor, warm_up_ratio, lr_min, lr_max, lr_max_decay:Union[float, str]) -> None:
        self.start          = 0
        self.end            = first_period
        self.period_factor  = period_factor
        self.lr_min         = lr_min
        self.lr_max         = lr_max
        self.lr_max_init    = lr_max
        self.lr_max_decay   = lr_max_decay
        self.warm_up_ratio  = warm_up_ratio
        self.max_epoch      = max_epoch
        self.period_num     = 1
    def step(self, current_epoch, optimizer):
        if current_epoch >= self.end:
            
            if self.lr_max_decay == "cosine":
                self.lr_max = self.lr_min + (1 / 2) * (self.lr_max_init - self.lr_min) * (1 + math.cos(
                        math.pi * (current_epoch) / (self.max_epoch)
                    )
                )
            else:
                self.lr_max = self.lr_max * self.lr_max_decay

            self.start = self.end
            self.end += int(self.end * self.period_factor)
            self.period_num += 1

        optimizer = warn_up_cosine_lr(
                start_epoch = self.start,
                num_epoch=current_epoch,
                warm_up_ratio=self.warm_up_ratio,
                optimizer=optimizer,
                max_epoch=self.end,
                lr_max=self.lr_max,
                lr_min=self.lr_min
            )

        return optimizer