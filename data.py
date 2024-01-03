
from PIL import Image
import os
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch
import pandas as pd
from transformers import BertTokenizerFast
import operator

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

def loo_split(anno_dict,subj_id):
    train_dict = dict()
    val_dict = dict()
    # Saliency4ASD dataset has 14 Controls and 14 ASDs
    if subj_id >=14: # ctrl
        subj_id -= 14
        cur_group = 1
    else:
        cur_group = 0 # asd

    group_name = ['asd','ctrl']

    for k in anno_dict.keys():
        if subj_id+1 > len(anno_dict[k][group_name[cur_group]]['fixation']):
            train_dict[k] = anno_dict[k] # current image does not have enough data, thus skipped for splitting
        else:
            train_dict[k] = dict()
            val_dict[k] = dict()
            # contructing data for the opposite group (no need for splitting)
            train_dict[k]['img_size'] = val_dict[k]['img_size'] = anno_dict[k]['img_size']
            train_dict[k][group_name[1-cur_group]] = anno_dict[k][group_name[1-cur_group]]
            val_dict[k][group_name[1-cur_group]] = dict()
            val_dict[k][group_name[1-cur_group]]['fixation'] = []
            val_dict[k][group_name[1-cur_group]]['duration'] = []

            # constructing data for the current group (split into train/val for leave-one-out validation)
            train_dict[k][group_name[cur_group]] = dict()
            val_dict[k][group_name[cur_group]] = dict()

            # splitting based on the relative position of the hold-out subjects
            if subj_id+1 == len(anno_dict[k][group_name[cur_group]]['fixation']):
                train_dict[k][group_name[cur_group]]['fixation'] = anno_dict[k][group_name[cur_group]]['fixation'][:subj_id]
                train_dict[k][group_name[cur_group]]['duration'] = anno_dict[k][group_name[cur_group]]['duration'][:subj_id]
            else:
                left_fix = anno_dict[k][group_name[cur_group]]['fixation'][:subj_id]
                right_fix = anno_dict[k][group_name[cur_group]]['fixation'][(subj_id+1):]
                if len(left_fix)>0 and not isinstance(left_fix[0],list):
                    left_fix = [left_fix]
                if len(right_fix)>0 and not isinstance(right_fix[0],list):
                    right_fix = [right_fix]

                left_dur = anno_dict[k][group_name[cur_group]]['duration'][:subj_id]
                right_dur = anno_dict[k][group_name[cur_group]]['duration'][(subj_id+1):]
                if len(left_dur)>0 and not isinstance(left_dur[0],list):
                    left_dur = [left_dur]
                if len(right_dur)>0 and not isinstance(right_dur[0],list):
                    right_dur = [right_dur]

                train_dict[k][group_name[cur_group]]['fixation'] = left_fix + right_fix
                train_dict[k][group_name[cur_group]]['duration'] = left_dur + right_dur

            val_dict[k][group_name[cur_group]]['fixation'] = [anno_dict[k][group_name[cur_group]]['fixation'][subj_id]]
            val_dict[k][group_name[cur_group]]['duration'] = [anno_dict[k][group_name[cur_group]]['duration'][subj_id]]

    return train_dict,val_dict



def image_selection(train_set,select_number=100):
    fisher_score = dict()
    for img in train_set.keys():
        asd_fix = train_set[img]['asd']['fixation']
        asd_dur = train_set[img]['asd']['duration']
        ctrl_fix = train_set[img]['ctrl']['fixation']
        ctrl_dur = train_set[img]['ctrl']['duration']
        img_size = train_set[img]['img_size']
        stat = [[] for _ in range(2)]

        # calculate the fisher score to select discriminative images
        for group, data in enumerate([(asd_fix,asd_dur),(ctrl_fix,ctrl_dur)]):
            cur_fix, cur_dur = data
            for i in range(len(cur_fix)):
                for j in range(len(cur_fix[i])):
                    y, x = cur_fix[i][j]
                    dist = np.sqrt((y-img_size[0]/2)**2 + (x-img_size[1]/2)**2)
                    dur = cur_dur[i][j]
                    stat[group].append([y,x,dist,dur])
        pos = np.array(stat[0])
        neg = np.array(stat[1])
        fisher = (np.mean(pos,axis=0)-np.mean(neg,axis=0))**2 / (np.std(pos,axis=0)**2 + np.std(pos,axis=0)**2) # fisher score
        fisher_score[img] = np.mean(fisher)

    # selecting the images by fisher score
    sorted_score = sorted(fisher_score.items(),key=operator.itemgetter(1))
    sorted_score.reverse()
    selected_img = []
    for i in range(select_number):
        selected_img.append(sorted_score[i][0])

    return selected_img

class ASDDataset(data.Dataset):
    def __init__(self,img_dir,data,valid_id,max_len,img_height,img_width,transform):
        self.img_dir = img_dir
        self.max_len = max_len
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self.initial_dataset(data, valid_id)
    
    def initial_dataset(self,data, valid_id):
        self.fixation = []
        self.duration = []
        self.label = []
        self.img_id = []
        self.img_size = []

        for img_id in data.keys():
            if not img_id in valid_id:
                continue
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
        valid_len = 0
        invalid = 0
        # only consider the first k fixations
        for i in range(self.max_len):
            if i+1 <= len(fixs):
                y_fix, x_fix = fixs[i]
                dur = durs[i]
                x_fix = int(x_fix*(self.img_width-1)/float(x_lim))
                y_fix = int(y_fix*(self.img_height-1)/float(y_lim))
                if x_fix >=0 and y_fix>=0 and x_fix <= self.img_width and y_fix <= self.img_height:
                    fixation.append([y_fix, x_fix]) # get the corresponding index of fixation on the downsampled feature map
                    duration.append(dur) # duration of corresponding fixation
                    valid_len += 1
                else:
                    invalid += 1
            else:
                fixation.append([0,0]) # pad if necessary
                duration.append(0)
        for i in range(invalid):
            fixation.append([0,0])
            duration.append(0)
        fixation = torch.from_numpy(np.array(fixation).astype('int'))
        duration = torch.from_numpy(np.array(duration).astype('int'))
        return fixation, duration, valid_len

    def __getitem__(self,index):
        img = Image.open(self.img_id[index])
        if self.transform is not None: 
            img = self.transform(img)
        label = torch.FloatTensor([self.label[index]])
        fixation, duration, valid_len = self.get_fix_dur(index)
        return img, label, fixation, duration, valid_len

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
    

    
