import os
from scipy.io import loadmat
import numpy as np
import glob
import json

def ivt_mod(coords, freq=240, min_dist=20, v_threshold=10):
    """
    Detects fixations from a series of raw gaze points (x,y) captured at particular frequency (freq) using the 
    velocity thresholding (v_threshold) method.

    Inspired from https://github.com/ecekt/eyegaze/blob/951f24d028fabca10cf329cd4c505a67c7377e6f/gaze.py#L16

    Arguments:
        coords      - coordinates (x,y) of gaze point on image
        freq        - frequency at which gaze points were captured (depends on the instrument used)
        min_dist    - minimum pixel distance between each fixation. If fixations lie closer than min_dist, they
                      are merged into a single fixation
        v_threshold - each gaze point having lesser velocity than v_threshold is considered as part of a fixation
    
    Returns:
        fixation - list of fixation coords [fy,fx]
        duration - list of durations for each fixation in ms
        
    """
    coords = np.array([c for c in coords if c[0] > 0 and c[1] > 0]) # removing ineligible gaze points
    x, y = coords[:,0], coords[:,1]
    
    # taking the index at which each gaze point was recorded as time
    times = np.array([ t for t in range(len(x)) ]) 

    difX = []
    difY = []
    tdif = []

    for i in range(len(x) - 1):
        difX.append(x[i+1] - x[i])
        difY.append(y[i+1] - y[i])
        tdif.append(times[i+1] - times[i])

    dif = np.sqrt(np.power(difX,2) + np.power(difY,2)) #in pix
    velocity = dif / tdif
    
    # First pass: Applying velocity thresholding to identify fixations
    # All of the fixation clusters are identified and then their centroid is considered as the fixation point
    fixs = []
    durs= []
    fs = []
    fx, fy, t0, r = x[0], y[0], 0, 0
    # print(f'starting from: ({fx:.2f}, {fy:.2f}) at t = {t0}')
    for i, v in enumerate(velocity):
        #print(f'\ni: {i} v: {v:.2f}')
        if v < v_threshold:
            # fixation
            # print(f'({x[i]:.2f}, {y[i]:.2f}) is a fixation',end=" | ")
            if r == 0:
                t0 = times[i]
            fx = (fx*r + x[i])/(r+1)
            fy = (fy*r + y[i])/(r+1)
            r += 1
            # print(f'after averaging: ({fx:.2f}, {fy:.2f}) r = {r}')
        else:
            # rint(f'({x[i]:.2f}, {y[i]:.2f}) is not a fixation')
            t1 = times[i]
            dur = t1 - t0
            if dur > 5:
                fixs.append([fy, fx])
                durs.append(dur)
                # print(f'appending fixation: ({fx:.2f}, {fy:.2f}) with duration: {dur:.2f}')
            fx, fy, t0, r = x[i], y[i], times[i], 0
    
    if len(fixs) == 0:
        return [], []
    # print(f'After first pass:\nfixs: {fixs}\ndurs: {durs}')
    # Second pass: Iterating through fixations and merging the fixations that are too close
    fixation = []
    duration = []
    fixy, fixx = fixs[0]
    # print(f'\nstarting from: ({fixx:.2f}, {fixy:.2f})')
    dur = 0
    r = 1
    for (fy, fx), t in zip(fixs, durs):
        # print(f'\nchecking ({fx:.2f}, {fy:.2f}) with dur = {t}')
        if abs(fixy - fy) < min_dist and abs(fixx - fx) < min_dist:
            # print('too close | ',end='')
            fixx = (fixx*r + fx)/(r+1)
            fixy = (fixy*r + fy)/(r+1)
            dur += t
            r += 1
            # print(f'after merging: ({fixx:.2f}, {fixy:.2f}) dur = {t}. r= {r}')
        else:
            if r != 1:
                # print(f'appending merged fixation: ({fixx:.2f}, {fixy:.2f}) with dur = {dur}')
                fixation.append([round(fixy,2), round(fixx,2)])
                duration.append(dur)
                
            # print(f'({fx:.2f}, {fy:.2f}) is not close to anyone, hence merging')
            fixation.append([round(fy, 2), round(fx)])
            duration.append(t)    
            
            fixy, fixx = fy, fx
            dur = 0
            r = 1
            
    if r == 0:
        fixation.append(fixs[-1])
        duration.append(durs[-1])
    
    duration = [round((float(t)/freq)*1000,2) for t in duration] # changing duration in ms
    return fixation, duration

# creating a view_dict = {person : [list of viewings (.mat files) by this person]}
root_dir = 'DATA/DATA/'
persons = [p for p in os.listdir(root_dir) if not (p.endswith('.mat') or p.endswith('.data') or p == '.DS_Store')]
view_dict = {p: os.listdir(os.path.join(root_dir, p)) for p in persons} 

# creating an anno dict where keys are the names of all stimuli images and value is a dictionary of fixations and durations
im_dir = 'ALLSTIMULI/ALLSTIMULI/'
anno_dict = {im[:-5]: {'fixations': [], 'durations': []} for im in os.listdir(im_dir)}

# iterating through all viewings by each person and extracting fixations and durations using ivt_mod() and storing it in anno_dict
# lots of try-except used to handle peculiar edge cases of this datatset
for person, matFiles in view_dict.items():
    for file in matFiles:
        im_name = file[:-4]
        mat_path = f'{root_dir}{person}/{file}'
        try:
            m = loadmat(mat_path)
        except:
            #print(f'unable to load {mat_path}')
            continue
        global coords
        try:
            coords = m[im_name][0][0][4][0][0][2]
        except:
            # image name is present partially in the matlab file [probable error from the creators of dataset]
            im_key = list(m.keys())[3] # key in matlab file
            matching_names = list(glob.glob(f'ALLSTIMULI/ALLSTIMULI/{im_key}*'))
            if len(matching_names) == 1: # if a single image matches im_key
                #print(f'sucessfully found a single im for {im_key}')
                try:
                    coords = m[im_key][0][0][4][0][0][2]
                except:
                    # another weird issue of different indexing for particular files
                    coords = m[im_key][0][0][0][0][0][2]
                    
            else:
                #print(f'unable to find a single im_name for {im_key}')
                #print(f'matched names: {matching_names}')
                continue
           
        fixation, duration = ivt_mod(coords, freq=240, min_dist=20, v_threshold=10)
        if len(fixation) != 0 and len(duration) != 0:
            anno_dict[im_name]['fixations'].append(fixation)
            anno_dict[im_name]['durations'].append(duration)

# cleaning annotations: removing im name keys for which no fixations were found
final_dict = {}
for im, d in anno_dict.items():
    fix = d['fixations']
    dur = d['durations']
    if len(fix) != 0:
        final_dict[im] = {'fixations': fix, 'durations': dur}

# saving annotations in a JSON file
with open('MIT1003_annotations.json', 'w') as fp:
    json.dump(final_dict, fp)