import numpy as np
import pandas as pd
import os
from glob import glob
import shutil
import random
import time
from datetime import datetime

time_now = str(datetime.now())
logfile_name = 'log_sort_he_image_{0}.txt'.format(time_now)
log_statement = 'This is to sort the he images in respective folders according to that particular fold indices\n'
f = open(logfile_name, 'w+')
f.write(log_statement)
message = ''

def load_labels():
    message = ''
    df = pd.read_excel('/dresden/users/aj611/experiments/biomed/AI_LiverBx_NAS_breakdown_20200206.xlsx', sheet_name='Sheet2')
    # deleter the column heading?
    df = df.iloc[:, 1:]
    
    # delete the 17th row i.e. patient 17_1, as we are using only 17_2.
    idx = 16
    df.drop(df.index[idx], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df)
    message += '{}'.format(df)

    # create a dictionary of the labels
    patient_labels = {'ID': list(df['STUDY_ID'].values),
            'Fibrosis': list(df['FIBROSIS'].values),
            'NAS-steatosis': list(df['NAS_STEATOSIS'].values),
            'NAS-lob': list(df['NAS_LOB_INFL'].values),
            'NAS-balloon': list(df['NAS_BALLOON'].values)}

    return patient_labels
    
# data directory for the HE images
data_dir='/dresden/users/aj611/experiments/biomed/feature_ensembling/all_patients_images_resized'     
#data_dir2='/dresden/gpu2/Datasets/Liver_Multi_Modality/Pathology/PNG_5x_224'                          
#data_dir2 = '/dresden/gpu2/Datasets/Liver_Multi_Modality/Pathology/PNG_5x_299_vsi'
data_dir2 = '/dresden/gpu2/Datasets/Liver_Multi_Modality/Pathology/PNG_15x_299_vsi'

fib_str = 'fib'
stea_str = 'nas_stea'
lob_str = 'nas_lob'
balloon_str = 'nas_balloon'

#output_path ='/dresden/users/aj611/experiments/biomed/he_images/'
#output_path ='/dresden/users/aj611/experiments/biomed/he_images_3x/'
#output_path ='/dresden/users/aj611/experiments/biomed/he_images_3x_oversample/'
output_path ='/dresden/users/aj611/experiments/biomed/he_test_prev/'
fib_base_path = output_path + fib_str
nas_stea_base_path = output_path + stea_str
nas_lob_base_path = output_path + lob_str
nas_balloon_base_path = output_path + balloon_str

folds = 3

# make the paths for all the data
def create_dirs(path, num_labels):
    new_path = ''
    for i in range(num_labels):
        for j in range(folds):
        # create the train, test dir
            for dir_str in ['test', 'train']:
                new_path = path + '/fold_{}/{}/{}'.format((j+1), dir_str, i) + '/'
                os.makedirs(new_path, exist_ok = True)
    
'''
fib_path = output_path + fib_str
fib_labels = 3

nas_stea_path = output_path + stea_str
nas_stea_labels = 2

nas_lob_path = output_path + lob_str
nas_lob_labels = 3

nas_balloon_path = output_path + balloon_str
nas_balloon_labels = 3

create_dirs(fib_path, fib_labels)
create_dirs(nas_stea_path, nas_stea_labels)
create_dirs(nas_lob_path, nas_lob_labels)
create_dirs(nas_balloon_path, nas_balloon_labels)
'''
for exp in ['fib', 'nas_stea', 'nas_lob', 'nas_balloon']:
    new_path = output_path + exp
    if exp == 'nas_stea':
        labels = 2
    else:
        labels = 3

    create_dirs(new_path, labels)

N = 32
patients = ['Patient_{}'.format(i) for i in range(1, N+1)]                                            
                                                                                                      
# find the patient folders present in the given pathi and
# put them into contents list
contents = []                                                                                         
for dirpath, dirs, files in os.walk(data_dir):                                                        
    for pat in patients:
        for dirpath in dirs:                                                                          
            if dirpath == pat:
                contents.append(pat)                                                                  
    
# get the number of patients for training and testing
total_num = len(contents)
train_ratio = 1/3
train_len = int(total_num * train_ratio )
test_len = int(total_num - train_len)
print(train_len)
print(test_len)
message += '\ntrain len {}'.format(train_len)
message += '\ntest len {}'.format(test_len)

# check for the entries present in patients list but not in contents list                             
# get the indices of tha vlaid patients
pat_idx = []
for pat in patients:
    if pat not in contents:                                                                           
        print('Absent', pat)                                                                          
        message += '\nAbsent {}'.format(pat)
    else:
        # the string looks like Patient_<id>, so splitting based on '_'
        cur_id = pat.split('_')
        # the last part of the split string contains the number, hence taking it
        cur_id = cur_id[-1]
        pat_idx.append(int(cur_id))

print('list of valid patients ', pat_idx)
message += '\nlist of valid patients {}'.format(pat_idx)


patient_labels = load_labels()


# create a dictionary of the train and test idxs
idx_dict = {}
random.seed(2)
random.shuffle(pat_idx)

N_fold = 3
l = int(np.ceil(len(pat_idx) / N_fold))

f.write(message)

# cross validation: fold i
for fold in range(N_fold):
    message = ''
    new_dict = {}
    fold_key = 'fold_{}'.format(fold+1)

    print('Fold {:d}'.format(fold+1))
    message += '\nFold {:d}'.format(fold+1)
    end = l*(fold+1) if l*(fold+1) <= len(pat_idx) else len(pat_idx)
    test_indices = pat_idx[l*(fold):end]
    train_indices = [idx for idx in pat_idx if idx not in test_indices]
    print('test indices', test_indices)
    print('train indices', train_indices)
    message += '\ntest indices {}'.format(test_indices)
    message += '\ntrain indices {}'.format(train_indices)

    new_dict['test_indices'] = test_indices
    new_dict['train_indices'] = train_indices
    idx_dict[fold_key] = new_dict 
print('\nlen of pat_idx :', len(pat_idx))
print('\npat_idx :', pat_idx)
message += '\nlen of pat_idx :{}'.format(len(pat_idx))
message += '\npat_idx :{}'.format(pat_idx)

f.write(message)

oversample_dict = {
        'fib':{'fold_2':['0_2x'], 'fold_3': ['0_3x', '1_2x']},
        'stea':{'fold_2':['0_2x'], 'fold_3':['0_2x']},
        'lob':{'fold_2':['0_2x']},
        'balloon':{'fold_1':['0_2x', '1_2x'], 'fold_2':['0_3x']}
        }

for fold in range(1, 4):
    # get the test_indices for this fold. test_indices contains the patient ids who are in test set
    # for the current fold
    cur_test_idx = []
    fold_str = 'fold_{}'.format(fold)
    cur_test_idx = idx_dict[fold_str]['test_indices']

    for j in range(len(pat_idx)):
        message = ''
        # patient ids are from 1, 32 whereas actual valid patient ids are from 0, 31 and total 30 in number
        # patient 15, 24 data had to be excluded
        i = pat_idx[j]
        fib_score = patient_labels['Fibrosis'][i-1]                                                
        nas_stea_score = patient_labels['NAS-steatosis'][i-1]                                          
        nas_lob_score = patient_labels['NAS-lob'][i-1]                                              
        nas_balloon_score = patient_labels['NAS-balloon'][i-1]                                        
        ID = patient_labels['ID'][i-1]

        print(f'i:{i}, ID:{ID},  fib:{fib_score}, stea:{nas_stea_score}, lob:{nas_lob_score}, balloon:{nas_balloon_score}')
        message += '\ni:{}, ID:{},  fib:{}, stea:{}, lob:{}, balloon:{}'.format(i, ID, fib_score, nas_stea_score, nas_lob_score, nas_balloon_score)


        if i == 17:                                                                                   
            pat_num = '{}_2'.format(i)                                                              
        else:
            pat_num = '{}'.format(i)
        HE_files = glob('{:s}/{:s}-HE*/*'.format(data_dir2, pat_num))
        #print(HE_files)

        # get the modified label according to our labeling
        if fib_score == 0:  # 0: 0
            fib_label = 0
        elif fib_score < 3:  # 1: [1, 2, 2.5]
            fib_label = 1
        else:               # 2: [3, 3.5, 4]
            fib_label = 2

        nas_stea_label = 0 if nas_stea_score < 2 else 1
        nas_lob_label = int(nas_lob_score) if int(nas_lob_score) < 2 else 2
        nas_balloon_label = int(nas_balloon_score)

        for k in range(len(HE_files)):
            full_src_path = HE_files[k]
            # th epath contains forward slash as part of the 
            # path, hence splitting it based on that, the
            # last part is the image name
            sub_str = full_src_path.split('/')
            img = sub_str[-1]
            path_str = ''
            if i not in cur_test_idx:
                path_str = '/train'
                print(f'{i} in train_set for fold {fold}')
                message += '\n{} in train_set for fold {}'.format(i, fold)
            else:
                path_str = '/test'
                print(f'{i} in test_set for fold {fold}')
                message += '\n{} in test_set for fold {}'.format(i, fold)
            # the same file needs to be copied into different folders for fibrosis and nas scores
            # creating the paths according to the labels
            full_dest_path_fib = fib_base_path + '/fold_{}'.format(fold) + path_str + '/{0}/'.format(int(fib_label)) + img
            full_dest_path_nas_stea = nas_stea_base_path + '/fold_{}'.format(fold) + path_str + '/{0}/'.format(int(nas_stea_label)) + img
            full_dest_path_nas_lob = nas_lob_base_path + '/fold_{}'.format(fold) + path_str + '/{0}/'.format(int(nas_lob_label)) + img
            full_dest_path_nas_balloon = nas_balloon_base_path + '/fold_{}'.format(fold) + path_str + '/{0}/'.format(int(nas_balloon_label)) + img
            
            # oversampling related

            # copy the files to the respective directories
            print(f'copying {full_src_path} to {full_dest_path_fib}\n')
            message += f'\ncopying {full_src_path} to {full_dest_path_fib}\n'
            shutil.copyfile(full_src_path, full_dest_path_fib)
            # check if the current fold requires oversampling

            print(f'copying {full_src_path} to {full_dest_path_nas_stea}\n')
            message += f'\ncopying {full_src_path} to {full_dest_path_nas_stea}\n'
            shutil.copyfile(full_src_path, full_dest_path_nas_stea)
            print(f'copying {full_src_path} to {full_dest_path_nas_lob}\n')
            message += f'\ncopying {full_src_path} to {full_dest_path_nas_lob}\n'
            shutil.copyfile(full_src_path, full_dest_path_nas_lob)
            print(f'copying {full_src_path} to {full_dest_path_nas_balloon}\n')
            message += f'\ncopying {full_src_path} to {full_dest_path_nas_balloon}\n'
            shutil.copyfile(full_src_path, full_dest_path_nas_balloon)
            if k == 10:
                break
        f.write(message)


f.close()
