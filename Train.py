import math
import torch
import datetime
from Loss import *
import numpy as np
from utils import *
import torch.nn as nn
from Model_v2 import LCT
from Dataset import *
import torchvision.models as models
from torch.utils.data import DataLoader

def train(model , train_dataloader, valid_dataloader, num_epochs, loss, opt):
    """
    Train a model

    * Input parameter:
    1. model                : model to be trained         
    2. train_dataloader     : dataloader of training dataset
    3. valid_dataloader     : dataloader of validation dataset
    4. num_epochs           : max # of epoch
    5. loss                 : loss function
    6. opt                  : optimizer 

    * Some Important parameter
    1. train_loss_log       : log of train loss
    2. valid_loss_log       : log of valid loss
    3. F1_score_log         : log of F1-score loss

    * Output:
    None
    """

    train_loss_log      = []
    valid_loss_log      = []
    F1_score_log        = []
    precision_log       = []
    recall_log          = []
    learning_rate_log   = []
    best_val_loss       = float('inf')
    record_datas        = {}

    loss_bce = loss
    
    optimizer = opt

    # get start time
    start_time = datetime.datetime.now()
    
    # train
    lr_scheduler = cosine_cycle_anneal_lr(
        max_epoch=num_epochs, 
        first_period=10, 
        period_factor=1.2, 
        warm_up_ratio=0.1, 
        lr_min=1e-7, 
        lr_max=1e-3, 
        lr_max_decay="cosine"
    )
    
    for epoch in range(num_epochs):
        optimizer = lr_scheduler.step(current_epoch=epoch, optimizer=optimizer)
        learning_rate_log.append(optimizer.param_groups[0]['lr'])
        print(f"epoch [{epoch+1:2d}/{num_epochs}]:\tlearning rate: {optimizer.param_groups[0]['lr']:4f}")

        train_loss_all = 0
        epoch_step = 0
        model.train()
        for i, data in enumerate(train_dataloader):
            window_value, label = data
            window_value, label = window_value.to(get_device()), label.to(get_device())

            optimizer.zero_grad()
            outputs = model(window_value)
            if torch.isnan(outputs).any():
                print(f"nan occur!")

            batch_loss = loss_bce(outputs, label)
            batch_loss.backward()
            optimizer.step()

            train_loss_all += batch_loss.item()
            epoch_step +=1
            processbar(
                now_process=i + 1, 
                all=len(train_dataloader), 
                total_len=30, 
                info=f"{i + 1:4d}/{len(train_dataloader):4d} batches of data have been trained.",
                needed_clear = True
            )

        
        train_avg_loss = train_loss_all / epoch_step
        train_loss_log.append(train_avg_loss)

        # validation
        model.eval()
        with torch.no_grad():
            valid_loss_all = 0
            val_step = 0
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i, data in enumerate(valid_dataloader):
                window_value, label = data
                window_value, label = window_value.to(get_device()), label.to(get_device())

                outputs = model(window_value)

                batch_loss = loss_bce(outputs, label)
                
                # get TP, FP, FN and F1-score
                _, preds = torch.max(outputs, dim=1)
                
                TP += (label * preds).sum()
                TN += ((1-label) * (1-preds)).sum()
                FP += ((1-label) * preds).sum()
                FN += (label * (1-preds)).sum()
                

                valid_loss_all += batch_loss.item()
                val_step +=1
                processbar(
                    now_process=i + 1, 
                    all=len(valid_dataloader), 
                    total_len=30, 
                    info=f"{i + 1:4d}/{len(valid_dataloader):4d} batches of data have been validated.",
                    needed_clear = True
                )
        
            valid_avg_loss = valid_loss_all / val_step
            valid_loss_log.append(valid_avg_loss)

        F1_score = TP / (TP + 1/2 * FP + 1/2 * FN)
        F1_score_log.append(F1_score)

        precision = TP / (TP + FP)
        precision_log.append(precision)

        recall = TP / (TP + FN)
        recall_log.append(recall)

        # print message
        print(f"Train loss: {train_avg_loss:.4f}, Valid loss: {valid_avg_loss:.4f}")
        print(f"F1-score  : {F1_score:.4f}, Precision : {precision:.4f}, Recall: {recall:.4f}")
        print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        
        # record data
        record_datas["train_loss_log"]      = train_loss_log
        record_datas["valid_loss_log"]      = valid_loss_log
        record_datas["F1-score_log"]        = F1_score_log
        record_datas["precision"]           = precision_log
        record_datas["recall"]              = recall_log
        record_datas["learning_rate_log"]   = learning_rate_log
        record_log(record_datas=record_datas)

        # save best model
        if valid_avg_loss < best_val_loss:
            best_val_loss = valid_avg_loss
            print("Saving better model...")
            torch.save(model.state_dict(), "./model/best_model.pth")
            print("Saving better model complete.")
            
        print("=================================================================")
        
        # save newest model
        torch.save(model.state_dict(), "./model/newest_model.pth")

    cost_time = datetime.datetime.now() - start_time
    print(f"Cost {cost_time} to training.")

def main():
    """
    Main function to train model

    * Input parameter: None

    * Some Important parameter
    1. batch_size       : batch size(hyper parameter)
    2. num_epochs       : max epoch (hyper parameter)
    3. model            : model     (hyper parameter)
    4. train_dataset    : training dataset
    5. train_dataloader : dataloader of training dataset
    4. valid_dataset    : validation dataset
    5. valid_dataloader : dataloader of validation dataset

    * Output:
    None
    """

    # hyper parameter
    batch_size = 300
    num_epochs = 106
    lr = 1e-7
    train_dataset_list = [1, 2, 3]
    valid_dataset_list = [23]
    #loss_wbce = nn.CrossEntropyLoss(weight=torch.Tensor([1., 300.]).to(get_device()))
    #loss_wbce = nn.CrossEntropyLoss(weight=torch.Tensor([1., 1.]).to(get_device()))
    loss_wbce = nn.CrossEntropyLoss()
    model = LCT(128)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=0.001)



    model.to(get_device())
    #model.to("cuda:1")
    #model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    model = nn.DataParallel(model, device_ids=[1, 2, 3])
    
    print("Loading training dataset...")
    train_dataset = multi_dataset(dataset_list = train_dataset_list, mode="train")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print("Loading training dataset complete.")

    print("Loading validation dataset...")
    valid_dataset = multi_dataset(dataset_list = valid_dataset_list, mode="valid")
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    print("Loading validation dataset complete.")

    print("Start training...")
    train(
        model=model,
        train_dataloader=train_dataloader, 
        num_epochs=num_epochs, 
        valid_dataloader=valid_dataloader, 
        loss=loss_wbce, 
        opt=optimizer
    )
    print("Training complete.")

if __name__ == "__main__":
    main()

### tmux Cirtl+b -> d