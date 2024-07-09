import torch

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

def step_lr(num_epoch, optimizer):
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
    sustain_epoch = 25

    lr = lr_inital * (lr_decay_factor ** (num_epoch // sustain_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
