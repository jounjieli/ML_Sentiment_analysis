
# coding: utf-8

from ML_utils import ML_utils
from ML_utils import ML_NLP_utlis
import shutil
import os
import logging
data_pos_dir_path = r'D:\ml_data\Data_set\NLP\sentiment_analysis\utf-8\2000\2000\pos'
data_neg_dir_path = r'D:\ml_data\Data_set\NLP\sentiment_analysis\utf-8\2000\2000\neg'
data_pos_list = ML_utils.merge_dir_file(data_pos_dir_path,save_name='merge_pos.txt',add_line_Feed=True,file_remove_LR=True)
data_neg_list = ML_utils.merge_dir_file(data_neg_dir_path,save_name='merge_neg.txt',add_line_Feed=True,file_remove_LR=True)
dir_path = r'D:\ml_data\GitHub\sentiment_analysis\Dataset'
shutil.move(os.path.join(data_pos_dir_path,'merge_pos.txt'), os.path.join(dir_path,'merge_pos.txt'))
shutil.move(os.path.join(data_neg_dir_path,'merge_neg.txt'), os.path.join(dir_path,'merge_neg.txt'))
ML_utils.dir_file_call_function(dir_path,ML_utils.file_remove_repeat_row, replace_old=True)
ML_utils.dir_file_call_function(dir_path, ML_NLP_utlis.file_to_opencc, replace_old=True)
ML_utils.dir_file_call_function(dir_path, ML_NLP_utlis.jieba_file_segmentation, replace_old=True)
