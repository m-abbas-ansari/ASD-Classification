
from PIL import Image
import os
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch
import pandas as pd
from transformers import BertTokenizerFast


def read_dataset(anno_path, start_idx=1, end_idx=301):
    anno_dict = dict()
    max_len = dict()
    # Saliency4ASD has 300 images
    for i in range(start_idx,end_idx):
        img = cv2.imread(os.path.join(anno_path,'Images',str(i)+'.png'))
        y_lim, x_lim, _ = img.shape
        anno_dict[i] = dict()
        anno_dict[i]['img_size'] = [y_lim,x_lim]
        asd = pd.read_csv(os.path.join(anno_path,'ASD','ASD_scanpath_'+str(i)+'.txt'))
        ctrl = pd.read_csv(os.path.join(anno_path,'TD','TD_scanpath_'+str(i)+'.txt'))
        group_name = ['ctrl','asd']
        for flag, group in enumerate([ctrl, asd]):
            anno_dict[i][group_name[flag]] = dict()
            anno_dict[i][group_name[flag]]['fixation'] = []
            anno_dict[i][group_name[flag]]['duration'] = []
            cur_idx = list(group['Idx'])
            cur_x = list(group[' x'])
            cur_y = list(group[' y'])
            cur_dur = list(group[' duration'])
            tmp_fix = []
            tmp_dur = []
            for j in range(len(cur_idx)):
                # finish loading data for one subject
                if cur_idx[j] == 0  and j != 0:
                    anno_dict[i][group_name[flag]]['fixation'].append(tmp_fix)
                    anno_dict[i][group_name[flag]]['duration'].append(tmp_dur)
                    tmp_fix = []
                    tmp_dur = []
                tmp_fix.append([cur_y[j],cur_x[j]])
                tmp_dur.append(cur_dur[j])
            # save data of the last subject
            anno_dict[i][group_name[flag]]['fixation'].append(tmp_fix)
            anno_dict[i][group_name[flag]]['duration'].append(tmp_dur)

    return anno_dict

class ASDDataset(data.Dataset):
    def __init__(self,img_dir,data,max_len,img_height,img_width,transform):
        self.img_dir = img_dir
        self.initial_dataset(data)
        self.max_len = max_len
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def initial_dataset(self,data):
        self.fixation = []
        self.duration = []
        self.label = []
        self.img_id = []
        self.img_size = []

        for img_id in data.keys():
            # if not img_id in valid_id:
            #     continue
            for group_label, group in enumerate(['ctrl','asd']):
                self.fixation.extend(data[img_id][group]['fixation'])
                self.duration.extend(data[img_id][group]['duration'])
                self.img_id.extend([os.path.join(self.img_dir,str(img_id)+'.png')]*len(data[img_id][group]['fixation']))
                self.label.extend([group_label]*len(data[img_id][group]['fixation']))
                self.img_size.extend([data[img_id]['img_size']]*len(data[img_id][group]['fixation']))

    def get_fix_dur(self,idx):
        fixs = self.fixation[idx]
        durs = self.duration[idx]
        y_lim, x_lim = self.img_size[idx]
        fixation = []
        duration = []
        invalid = 0
        # only consider the first k fixations
        for i in range(self.max_len):
            if i+1 <= len(fixs):
                y_fix, x_fix = fixs[i]
                dur = durs[i]
                x_fix = int(x_fix*(self.img_width/float(x_lim))/32)
                y_fix = int(y_fix*(self.img_height/float(y_lim))/33)
                if x_fix >=0 and y_fix>=0:
                    fixation.append(y_fix*25 + x_fix) # get the corresponding index of fixation on the downsampled feature map
                    duration.append(dur) # duration of corresponding fixation
                else:
                    invalid += 1
            else:
                fixation.append(0) # pad if necessary
                duration.append(0)
        for i in range(invalid):
            fixation.append(0)
            duration.append(0)
        fixation = torch.from_numpy(np.array(fixation).astype('int'))
        duration = torch.from_numpy(np.array(duration).astype('int'))
        return fixation, duration

    def __getitem__(self,index):
        img = Image.open(self.img_id[index])
        if self.transform is not None:
            img = self.transform(img)
        label = torch.FloatTensor([self.label[index]])
        fixation, duration = self.get_fix_dur(index)
        return img, label, fixation, duration

    def __len__(self,):
        return len(self.fixation)


class CaptionDataset(data.Dataset):
    def __init__(self, cap_data, max_len,
                 visual=False, fix_data=None, crop_radius=85,
                 im_dir="TrainingDataset/TrainingData/Images/",
                 transform=None):
        self.transform = transform
        self.visual = visual
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', padding_side='right',
                                                      truncation_side='right')

        self.fix_crops = []
        self.fix_tokens = []
        self.label = []
        for img_id in cap_data.keys():
            if visual:
                img = np.array(Image.open(im_dir + f"{img_id}.png"))
                h, w, _ = img.shape
                pad = crop_radius
                canvas = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=img.dtype)
                canvas[pad:-pad, pad:-pad, :] = img

            for group_label, group in enumerate(['ctrl', 'asd']):
                for per_caps in cap_data[img_id][group]:
                    while len(per_caps) < max_len:
                        per_caps.append("")  # adding empty captions as padding for fix lens < max len
                    tokens = tokenizer(per_caps[:max_len], max_length=128, padding='max_length',
                                       truncation=True, return_tensors='pt')
                    self.fix_tokens.append(tokens)
                    self.label.append(group_label)

                if visual:
                    for fix_cords in fix_data[img_id][group]['fixation']:
                        crops = []
                        for fix in fix_cords:
                            cropi = canvas[fix[0]: fix[0] + 2 * pad, fix[1]: fix[1] + 2 * pad, :]
                            crops.append(cropi)
                        while len(crops) < max_len:
                            crops.append(np.zeros((pad * 2, pad * 2, 3), dtype=img.dtype))
                        self.fix_crops.append(crops[:max_len])

    def __getitem__(self, index):
        label = torch.FloatTensor([self.label[index]])
        fix_tokens = self.fix_tokens[index]
        if not self.visual:
            return fix_tokens, label
        else:
            crops = self.fix_crops[index]
            if self.transform is not None:
                crops = [self.transform(c) for c in crops]
            return crops, fix_tokens, label

    def __len__(self, ):
        return len(self.label)


if __name__ == '__main__':
    anno_dir = '../../Datasets/Saliency4ASD/TrainingData'
    train_anno = read_dataset(anno_dir, 1, 241)
    val_anno = read_dataset(anno_dir, 241, 301)
    print(len(train_anno))
    print(len(val_anno))
    # Testing data loader
    img_dir = '../../Datasets/Saliency4ASD/TrainingData/Images'
    max_len = 14
    img_height = 600
    img_width = 800
    transform = transforms.Compose([transforms.Resize((img_height,img_width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_set = ASDDataset(img_dir,train_anno,max_len,img_height,img_width,transform)
    val_set = ASDDataset(img_dir,val_anno,max_len,img_height,img_width,transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=12, shuffle=True, num_workers=2)
    vallodaer = torch.utils.data.DataLoader(val_set, batch_size=12, shuffle=False, num_workers=2)
    for i, data in enumerate(trainloader):
        img, label, fixation, duration = data
        print(img.shape)
        print(label.shape)
        print(fixation.shape)
        print(duration.shape)
        break
    for i, data in enumerate(vallodaer):
        img, label, fixation, duration = data
        print(img.shape)
        print(label.shape)
        print(fixation.shape)
        print(duration.shape)
        break
    

    
