
# coding: utf-8

# In[1]:


from ML_utils import ML_utils
import shutil
import os
import numpy as np
from gensim.models import word2vec
import pandas as pd
#從Dateset合併資料
data_pos_dir_path = r'D:\Backup\ml_data\Data_set\NLP\sentiment_analysis\utf-8\6000\6000\pos'
data_neg_dir_path = r'D:\Backup\ml_data\Data_set\NLP\sentiment_analysis\utf-8\6000\6000\neg'
ML_utils.Merge_dir_file(data_pos_dir_path,save_name='merge_hotel_pos.txt',regular=True,file_filter_='^pos',
                                        add_line_Feed=True,file_remove_LR=True)
ML_utils.Merge_dir_file(data_neg_dir_path,save_name='merge_hotel_neg.txt',regular=True,file_filter_='^neg',
                                        add_line_Feed=True,file_remove_LR=True)
ML_utils.Merge_dir_file(data_pos_dir_path,save_name='merge_pos.txt',regular=True,file_filter_='^merge',
                                        add_line_Feed=False,file_remove_LR=False)
ML_utils.Merge_dir_file(data_neg_dir_path,save_name='merge_neg.txt',regular=True,file_filter_='^merge',
                                        add_line_Feed=False,file_remove_LR=False)
#移至專案
dir_path = r'D:\Backup\ml_data\GitHub\ML_Sentiment_analysis\Dataset'
shutil.move(os.path.join(data_pos_dir_path,'merge_pos.txt'), os.path.join(dir_path,'merge_pos.txt'))
shutil.move(os.path.join(data_neg_dir_path,'merge_neg.txt'), os.path.join(dir_path,'merge_neg.txt'))
#前處理
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Remove_file_repeat_row, replace_old=True)
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Opencc_file, replace_old=True)
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Jieba_file_segmentation, replace_old=True)
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Filter_file_wlen,replace_old=False, regular=True,file_filter_='^merge',
                                max_word_num=60,file_head_name='cut_')
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Shuffle_file,replace_old=True, regular=True,file_filter_='^cut_')
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Trim_file_rows,replace_old=True,
                                file_filter_='cut_',row_num=2000)
sentiment_2_daat = ML_utils.Merge_dir_file(dir_path,save_name='sentiment_2_daat.txt',
                                           add_line_Feed=False,file_remove_LR=False,file_filter_='cut_')


# In[2]:


#轉成list，長度不足填補0
x = ML_utils.CallF_DirFile(dir_path,ML_utils.WordToList_file,max_word_num=60,padding=0,file_filter_='sentiment')[0]
#載入word2vec模型
model_path = r'D:\Backup\ml_data\GitHub\ML_Sentiment_analysis\word2vec_model\word2vec_model'
model = word2vec.Word2Vec.load(model_path)
#vec_padding
vec_padding = np.zeros((1,300),np.float32)
ML_utils.ToVec_list_save(x,os.path.join(dir_path,'vec_x_neg2000_pos2000'),
                         model,vec_padding,word_padding=0,word_padding_vec=vec_padding)
y = np.concatenate( (np.full(2000,0),np.full(2000,1)) )
y = pd.Series(y)
y_one_hot_encoding = pd.get_dummies(y)
y_one_hot_encoding = y_one_hot_encoding.to_numpy().astype(np.float32)
np.save(os.path.join(dir_path,'vec_y_neg2000_pos2000'),y_one_hot_encoding)

