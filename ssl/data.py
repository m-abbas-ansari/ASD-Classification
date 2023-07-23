
import PIL
from PIL import Image
import json
import numpy as np
import torch.utils.data as data
# from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from random import sample
import torch
import os
import operator
import cv2
import pandas as pd
import random

def read_dataset(anno_dir, datasets, val_ratio = 0.2):
    """
    We split each dataset in datasets list into train and val
    """
    train_anno = {'fixation': [],
                  'duration': [],
                  'img_id': [],
                  'dva': [],
                  'img_size': []}
    
    val_anno = {'fixation': [],
                'duration': [],
                'img_id': [],
                'dva': [],
                'img_size': []}
    
    ret_anno = (train_anno, val_anno)
    
    for data in datasets:
        anno = {}
        with open(f"{anno_dir}/ANNOTATIONS/{data}.json", 'r') as f:
            anno = json.load(f)
        ims = list(anno.keys())
        val_len = int(val_ratio * len(ims))
        val_ims = sample(ims, val_len)
        train_ims = list(set(ims) - set(val_ims))
        
        for i, g in enumerate((train_ims, val_ims)):
            for im in g:
                num = len(anno[im]['fixations'])
                ret_anno[i]['fixation'].extend(anno[im]['fixations'])
                if 'durations' in anno[im]:
                    ret_anno[i]['duration'].extend(anno[im]['durations'])
                else:
                    ret_anno[i]['duration'].extend([0]*num)
                    
                ret_anno[i]['img_id'].extend([f"{anno_dir}/STIMULI/{data}/{im}.jpg"]*num)
                ret_anno[i]['dva'].extend([anno[im]['dva']]*num)
                ret_anno[i]['img_size'].extend([anno[im]['img_size']]*num)
    
    return ret_anno

def read_ASD_dataset(anno_path, start_idx=1, end_idx=301):
    anno_dict = dict()
    max_len = dict()
    # Saliency4ASD has 300 images
    for i in range(start_idx,end_idx):
        img = Image.open(os.path.join(anno_path,'Images',str(i)+'.png'))
        y_lim, x_lim = img.size[::-1]
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

class FixDataset(data.Dataset):
    def __init__(self, data, max_len=20, img_height=320, img_width=512, transform=None):
        
        self.img_size = data['img_size']
        self.fixation = data['fixation']
        self.img_height = img_height
        self.img_width = img_width
        self.fixation = [self.scale_fixations(self.fixation[i], self.img_size[i]) 
                            for i in range(len(self.img_size))]
        self.duration = data['duration']
        
        self.img_id = data['img_id']
        self.dva = data['dva']
        self.max_len = max_len
       
        self.transform = transform
    
    def scale_fixations(self, fixs, orig_size):
        H, W = orig_size
        h, w = self.img_height, self.img_width 
        scaled_fix = []
        for fix in fixs:
            fy = int(fix[0]*(h-1)/H)
            fx = int(fix[1]*(w-1)/W)
            if fy > 320 or fx > 512:
                continue
            scaled_fix.append([fy, fx])
        
        return scaled_fix

    def __getitem__(self, index):
        img = Image.open(self.img_id[index])
        if self.transform is not None:
            img = self.transform(img)
            
        fix = self.fixation[index]
        dur = self.duration[index]
        dva = self.dva[index]
        img_size = self.img_size[index]

        num_fix = len(fix)
        padding_mask = torch.tensor([0 for _ in range(160 + self.max_len)]).to(torch.bool)
        padding_mask[160 + num_fix:] = True
        while len(fix) < self.max_len:
            fix.append([0,0])
            #dur.append(0)
            
        fixation = torch.from_numpy(np.array(fix[:self.max_len]).astype('int'))
        #duration = torch.from_numpy(np.array(dur[:self.max_len]).astype('int'))
        
        return img, fixation, padding_mask
    
    
    def __len__(self, ):
        return len(self.fixation)

class FixViewDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=20, img_height=320, img_width=512, augmentations=[],transform=None):
        self.img_height = img_height
        self.img_width = img_width
        self.fixation = data['fixation']
        self.img_id = data['img_id']
        self.augmentations = augmentations
        self.max_len = max_len
        self.transform = transform
        self.augmentation_map = {
            'noise_addition': self.noise_addition,
            'horizontal_flip': self.horizontal_flip,
            'segment_deletion': self.segment_deletion,
            'scanpath_reversal': self.scanpath_reversal,
            'global_rotation_w_par_fix': self.global_rotation_w_par_fix,
            'global_rotation_only_img': self.global_rotation_only_img,
            'global_rotation_w_fix': self.global_rotation_w_fix,
        }
    
    def scale_fixations(self, fixs, orig_size, new_size):
        H, W = orig_size
        h, w = new_size
        
        scaled_fix = []
        for fix in fixs:
            fy = int(fix[0]*(h-1)/H)
            fx = int(fix[1]*(w-1)/W)
            if fy > h or fx > w:
                continue
            scaled_fix.append([fy, fx])
        
        return scaled_fix
    
    def noise_addition(self, image, fixations, mean=0, std=25):
        noise = np.random.uniform(mean, std, size=(len(fixations), 2))
        new_fixations = np.add(fixations, noise).tolist()
        return image, new_fixations

    def horizontal_flip(self, image, fixations):
        new_image = PIL.ImageOps.mirror(image)
        w, h = image.size
        new_fixations = [[f[0], w - f[1]] for f in fixations]
        return new_image, new_fixations

    def segment_deletion(self, image, fixations, dropout_prob=0.20):
        new_fixations = []
        for fix in fixations:
            if random.random() > dropout_prob:
                new_fixations.append(fix)
        return image, new_fixations

    def scanpath_reversal(self, image, fixations):
        new_fixations = fixations[::-1]
        return image, new_fixations

    def get_rotated_fix(self, w, h, angle, fixations):
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        new_fix = []
        for fix in fixations:
            v = [fix[1],fix[0],1]
            # Perform the actual rotation and return the image
            calculated = np.dot(M,v)
            new_fix.append([calculated[1], calculated[0]])
            
        return new_fix
    
    def global_rotation_w_fix(self, image, fixations):
        angle = random.randint(-30, 30)
        w, h = image.size
        new_image = image.rotate(angle, expand=True)
        new_fix = self.get_rotated_fix(w,h, angle, fixations)
        
        return new_image, new_fix
    
    def global_rotation_only_img(self, image, fixations):
        angle = random.randint(-30, 30)
        new_image = image.rotate(angle, expand=True)
        #new_fix = self.scale_fixations(fixations, image.size[::-1], new_image.size[::-1])
        return new_image, fixations
    
    def global_rotation_w_par_fix(self, image, fixations):
        angle = random.randint(-30, 30)
        w, h = image.size
        if angle > 0:
            fix_angle = random.randint(0, angle)
        else:
            fix_angle = random.randint(angle, 0)
        new_image = image.rotate(angle, expand=True)
        new_fix = self.get_rotated_fix(w, h, fix_angle, fixations)
        return new_image, new_fix
    
    def get_view(self, img, fix, i):
        for aug in self.augmentations: # apply specified augmentations
            if aug == 'horizontal_flip' or aug == 'scanpath_reversal':
                if i == 1:
                    img, fix = self.augmentation_map[aug](img, fix)
            else:    
                img, fix = self.augmentation_map[aug](img, fix)

        fix = self.scale_fixations(fix, img.size[::-1], (self.img_height, self.img_width))
        
        if self.transform is not None:
            img = self.transform(img)
            
        num_fix = len(fix)
        padding_mask = torch.tensor([0 for _ in range(160 + self.max_len)]).to(torch.bool)
        padding_mask[160 + num_fix:] = True
        while len(fix) < self.max_len:
            fix.append([0,0])            
        fixation = torch.from_numpy(np.array(fix[:self.max_len]).astype('int'))
        
        return img, fixation, padding_mask
        
    def __getitem__(self, index):
        img = Image.open(self.img_id[index])         
        fix = np.array(self.fixation[index].copy())
        
        return self.get_view(img.copy(), fix.copy(), 0), self.get_view(img.copy(), fix.copy(), 1) 
    
    def __len__(self, ):
        return len(self.fixation)

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

class ASDHATDataset(data.Dataset):
    def __init__(self, img_dir, data, valid_id, max_len=20, img_height=320, img_width=512, transform=None):
        self.img_dir = img_dir
        self.max_len = max_len
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self.initial_dataset(data, valid_id)
    
    def scale_fixations(self, fixs, orig_size):
        H, W = orig_size
        h, w = self.img_height, self.img_width
        
        scaled_fix = []
        for fix in fixs:
            fy = int(fix[0]*(h-1)/H)
            fx = int(fix[1]*(w-1)/W)
            if fy > h or fx > w:
                continue
            scaled_fix.append([fy, fx])
        
        return scaled_fix
    
    def initial_dataset(self, data, valid_id):
        self.fixation = []
        self.duration = []
        self.label = []
        self.img_id = []
        
        for img_id in data.keys():
            if not img_id in valid_id:
                continue
            for group_label, group in enumerate(['ctrl', 'asd']):
                fixs = [self.scale_fixations(fs, data[img_id]['img_size']) 
                            for fs in data[img_id][group]['fixation']]
                durs = data[img_id][group]['duration']
                img_path = os.path.join(self.img_dir,str(img_id)+'.png')
                
                self.fixation.extend(fixs)
                self.duration.extend(durs)
                self.img_id.extend([img_path]*len(fixs))
                self.label.extend([group_label]*len(fixs))
    
    def __getitem__(self, index):
        img = Image.open(self.img_id[index])
        if self.transform is not None:
            img = self.transform(img)
            
        fix = self.fixation[index]
        dur = self.duration[index]
        label = torch.LongTensor([self.label[index]])
        num_fix = len(fix)
        padding_mask = torch.tensor([0 for _ in range(160 + self.max_len)]).to(torch.bool)
        
        padding_mask[160 + num_fix:] = True
        while len(fix) < self.max_len:
            fix.append([0,0])
            dur.append(0)
            
        fixation = torch.from_numpy(np.array(fix[:self.max_len]).astype('int'))
        duration = torch.from_numpy(np.array(dur[:self.max_len]).astype('int'))
        
        return img, fixation, duration, padding_mask, label
    
    
    def __len__(self, ):
        return len(self.fixation)
    
'''
class FixTensorDatasetCPU(data.Dataset):
    def __init__(self, data_dir, im_size = (320, 512), mask_ratio=0.2, hidden_dim=256):
        self.im_size = im_size
        self.hidden_dim = hidden_dim
        self.device = torch.device("cpu")
        self.mask_ratio = mask_ratio
        self.fixations = [os.path.join(data_dir, f"fix-{i}.pt") for i in range(28921)]
        self.pad_masks = [os.path.join(data_dir, f"pad-{i}.pt") for i in range(28921)]
        self.perceptual_features = [os.path.join(data_dir, f"p1-{i}.pt") for i in range(28921)]
        self.foveal_features = [os.path.join(data_dir, f"p4-{i}.pt") for i in range(28921)]
        
        self.pos_embs = PositionEmbeddingSine(hidden_dim // 2, 
                                              normalize=True
                                              )(torch.rand((1, 3, 320, 512))).flatten(2).to(self.device)
        self.per_pos_embs = self.get_per_pos_embs(self.pos_embs, (320, 512), (10, 16))
        
    def get_per_pos_embs(self, pos_emb, im_size, feat_size):
        H, W = im_size
        h, w = feat_size
        center_coords = torch.LongTensor([int(H/(2*h)*(2*y + 1)) * W + int(W/(2*w)*(2*x + 1)) 
                                      for y in range(h) 
                                          for x in range(w)])
            
        bs, dim, _ = pos_emb.size()
        coords = center_coords.expand(bs, dim, center_coords.size()[0])
        x = pos_emb.gather(2, coords)
        
        return x
        
    def get_fix_tokens(self, x, fixs):
        H, W = self.im_size
        _, feat, h, w = x.size()
        
        # We get fixation index in downscaled feature map for given fix coords
        cint64 = torch.torch.cuda.LongTensor
        fixation = (fixs[:, :, 0]*(h/H)).long()*w + (fixs[:, :, 1]*(w/W)).long()
        #print(fixs.get_device(), fixation.get_device())
        x = x.flatten(2)
        fixation = fixation.view(fixation.size(0), 1, fixation.size(1))
        fixation = fixation.expand(fixation.size(0), feat, fixation.size(2))
        x = x.gather(2, fixation)
        
        return x

    def get_fov_pos_embs(self, pos_emb, fixs):
        H, W = self.im_size
        
        # flatten fix coords on original im_size
        cint64 = torch.torch.cuda.LongTensor
        fixation = (fixs[:,:, 0]*W + fixs[:,:, 1]).long()
        
        bs, l = fixation.size()
        _, feat, num = pos_emb.size()
        pos_emb = pos_emb.expand(bs, feat, num)
        fixation = fixation.view(bs, 1, l)
        fixation = fixation.expand(bs, feat, l)
        x = pos_emb.gather(2, fixation)
        
        return x
    
    def random_mask_tokens(self, fov_tokens, padding_mask):
        batch_size, dim, seq_length = fov_tokens.size()
        
        num_fix = int(20 - padding_mask.sum()) # find number of fixations [excluding padding]
        # Calculate the number of tokens to mask per batch
        num_tokens_to_mask = int(num_fix * self.mask_ratio)
        if num_tokens_to_mask == 0:
            num_tokens_to_mask = 1 # atleast mask one token

        # Generate random mask positions for each batch
        mask_positions = torch.randint(num_fix, (batch_size, num_tokens_to_mask))
        
        # Create the mask tensor
        mask = torch.zeros((batch_size, seq_length), dtype=torch.bool)
        # Set 0s at the randomly selected positions
        for i in range(batch_size):
            mask[i, mask_positions[i]] = 1
        # Broadcast the mask to match the dimensions of fov_tokens
        mask = mask.unsqueeze(1).expand(-1, dim, -1)
        # Apply the mask to fov_tokens
        fov_tokens_masked = fov_tokens.masked_fill(mask, 0)
        
        return fov_tokens_masked
    
    
    def __getitem__(self, index):
        fixs = torch.load(self.fixations[index]).unsqueeze(0)
        padding_mask = torch.load(self.pad_masks[index])
        p1 = torch.load(self.perceptual_features[index], map_location=self.device).unsqueeze(0)
        per_tokens = p1.flatten(2)
        p4 = torch.load(self.foveal_features[index], map_location=self.device).unsqueeze(0)
        fov_tokens = self.get_fix_tokens(p4, fixs)
        
        fov_pos_embs = self.get_fov_pos_embs(self.pos_embs, fixs) 
        per_tokens = per_tokens + self.per_pos_embs
        fov_tokens = fov_tokens + fov_pos_embs
        
        fov_tokens = self.random_mask_tokens(fov_tokens, padding_mask) # mask out certain foveal tokens (These are to be predicted by the model)

        return fixs.squeeze(0), padding_mask, per_tokens.squeeze(0), fov_tokens.squeeze(0)
    
    def __len__(self,):
        return len(self.fixations)
'''
