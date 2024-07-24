import os
import re
import math
import torch
from utils import processbar
import numpy as np
from imblearn.over_sampling import SMOTE
from pyedflib import highlevel
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class EDFdataset(Dataset):
    def __init__(
            self,
            freq:int = 256,
            dataset_folder_path:str = "",
            duration:int = 1,
            overlay:float = 0.25,
            mode = ""
        ) -> None:
        """
        Constructor of class of EDFDataset

        * Input parameter:
        1. freq                 : edf file sample freqency         
        2. dataset_folder_path  : directory of dataset folder
        3. duration             : single window duration (unit: second)     
        4. overlay              : every window's overlay
        
        * Self parameter
        1. self.freq                : edf file sample freqency         
        2. self.dataset_folder_path : directory of dataset folder
        3. self.duration            : single window duration (unit: second)     
        4. self.overlay             : every window's overlay
        5. self.summary_dictory     : a dictory record every summary of files
        6. self.dataset_name        : name of dataset
        7. self.trasfroms           : image transform
        8. self.datas               : datas. List of tuple (value, label)

        * Output:
        None
        """
        super().__init__()      
        self.freq                   = freq         
        self.dataset_folder_path    = dataset_folder_path        
        self.duration               = duration         
        self.overlay                = overlay
        self.summary_dictory        = []
        self.dataset_name           = self.dataset_folder_path.split("/")[-2]
        self.datas                  = []
        self.mode                   = mode
        self.transform              = transforms.Compose([
            transforms.Normalize((.5), (.5))
        ])

        assert self.mode.lower() == "train" or self.mode.lower() == "valid"

        self.features = []
        self.label = []

        # read summary file and create dircetory
        self.summary_files()

        # split windows 
        self.split_windows()

        # balance
        if self.mode.lower() == "train":
            self.balance()
            

    def summary_files(self) -> None:
        """
        process summary file and create dictory record summary of each files
        """
        # get summary file name
        summary_file = [string for string in os.listdir(self.dataset_folder_path) if 'summary' in string]
        
        # open it and ignore header
        with open(self.dataset_folder_path + summary_file[0], "r") as f:
            self.summary_str = f.read().split("\n\n")[2:]

        # create dictory for every summary of single file
        for single_file_summary in self.summary_str:
            single_dictory = {}
            
            # find keys and values with regex
            keys = re.findall(r"(.*): ", single_file_summary)
            values = re.findall(r": (.*[^ seconds&^\n])", single_file_summary)
            
            # correct some data type to be `int`
            values = [int(value) if idx >=3 else value for idx, value in enumerate(values)]

            # create single dictory
            for key_value in zip(keys, values):
                single_dictory[key_value[0]] = key_value[1]
            
            # append to self.summary_dictory
            self.summary_dictory.append(single_dictory)

    def split_windows(self)->None:
        edf_files = [single_dictionary["File Name"] for single_dictionary in self.summary_dictory]
        edf_files = [self.dataset_folder_path + edf_file for edf_file in edf_files]

        # Create windows data and label with summary
        for which_file, (file, summary) in enumerate(zip(edf_files, self.summary_dictory)):
            # read edf file
            signals, signal_headers, header = highlevel.read_edf(file)

            # size: [H, W] -> [1, H, W] (channel = 1, gray image)
            signals_tensor = torch.Tensor(signals).unsqueeze(0)

            # some edf file contain 24th signal (ECG), clean it
            signals_tensor = signals_tensor[:, :23, :] if signals_tensor.shape[1] > 23 else signals_tensor

            # standardization each channel
            signals_tensor = (signals_tensor - signals_tensor.mean(dim=2, keepdim=True)) / signals_tensor.std(dim=2, keepdim=True)
            
            #signals_tensor = torch.nn.functional.normalize(signals_tensor, dim=2)

            # split windows
            window_width = self.duration * self.freq
            start_point = 0
            end_point = window_width
            while end_point <= signals_tensor.shape[2]:
                # declare label
                label = 0
                summary_list = list(summary.values())

                # get seizures_intervals
                seizures_intervals = summary_list[4:]
                
                # check if this window in seizures_interval
                for idx in range(len(seizures_intervals)//2):
                    seizures_intervals_start    = seizures_intervals[idx * 2] * self.freq
                    seizures_intervals_end      = seizures_intervals[idx * 2 + 1] * self.freq
                    if (
                        (seizures_intervals_start <= start_point and start_point <= seizures_intervals_end) or 
                        (seizures_intervals_start <= end_point and end_point <= seizures_intervals_end)
                    ):
                        label = 1
                    else:
                        label = 0
                # append data
                self.datas.append((signals_tensor[:, :, start_point : end_point], label))
                self.features.append(signals_tensor[:, :, start_point: end_point].squeeze(0).numpy())
                self.label.append(label)

                # move window
                start_point += int((1 - self.overlay) * window_width)
                end_point += int((1 - self.overlay) * window_width)
            
            processbar(which_file+1, len(edf_files), total_len = 30, info = f"{which_file + 1:2d}/{len(edf_files):2d} edf file processed.")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return (self.transform(self.datas[index][0]), self.datas[index][1])
    
    def append_other_dataset(self, datasets_list):
        """
        self dataset = self dataset + a list of other datasets.
        """
        for single_dataset in datasets_list:
            self.datas += single_dataset.datas
    
    def balance(self):
        X = []
        Y = []
        for i in range(np.array(self.features).shape[2]):
            X_res, Y_res = SMOTE(random_state=42).fit_resample(np.array(self.features)[:,:,i], np.array(self.label))
            X.append(torch.tensor(X_res).unsqueeze(2))
            Y = torch.tensor(Y_res)

            processbar(
                now_process=i + 1, 
                all=np.array(self.features).shape[2], 
                total_len=30, 
                info=f"{i + 1}/{np.array(self.features).shape[2]} balanced part1.",
                needed_clear = True
            )
        X = torch.cat(X,2)
        C = []
        for i in range(X.shape[0]):
            C.append((X.unsqueeze(1)[i,:,:,:], Y[i]))

            processbar(
                now_process=i + 1, 
                all=X.shape[0], 
                total_len=30, 
                info=f"{i + 1}/{X.shape[0]} balanced part2.",
                needed_clear = True
            )
        self.datas = C.copy()


def multi_dataset(
    freq:int = 256,
    dataset_list:list = [1],
    duration:int = 1,
    overlay:float = 0.25,
    mode:str = ""
) -> EDFdataset:
    print(f"loading ", end="")
    dataset_name = ["chb" + (str(single_dataset).zfill(2)) for single_dataset in dataset_list]
    for single_dataset_name in dataset_name:
        if single_dataset_name == dataset_name[-1]:
            print(single_dataset_name + " ", end="")
        else:
            print(single_dataset_name + ", ", end="")
    print(f"datasets...")

    multi_dataset = []
    # create multi dataset
    for single_dataset in dataset_list:
        dataset_folder_path = f"/homes/nfs/caslab_bs/Desktop/Dennis/physionet.org/files/chbmit/1.0.0/chb{str(single_dataset).zfill(2)}/"
        multi_dataset.append(
            EDFdataset(
                freq=freq,
                dataset_folder_path=dataset_folder_path,
                duration=duration,
                overlay=overlay,
                mode=mode
            )
        )

    # combine each dataset
    first_dataset = multi_dataset[0]
    first_dataset.append_other_dataset(multi_dataset[1:])

    return first_dataset

#a = EDFdataset(dataset_folder_path="/homes/nfs/caslab_bs/Desktop/Dennis/physionet.org/files/chbmit/1.0.0/chb01/")
# dataLoader = DataLoader(dataset=a, batch_size=512, shuffle=True)
# for i, data in enumerate(dataLoader):
#     window_value, label = data
#     print(window_value[0])
#     print(window_value[1])
#     break

# neg_num = 0
# pos_num = 0
# 
# for idx, data in enumerate(a):
#     if data[1] == 1 :
#         pos_num += 1
#     else:
#         neg_num += 1
#     processbar(idx+1, len(a), total_len = 30, info = f"{idx + 1}/{len(a)} windows scanned.")
# 
# print(f"pos: {pos_num}")
# print(f"neg: {neg_num}")
# print(f"alpha: {neg_num/ pos_num}")
