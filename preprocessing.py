
# coding: utf-8

# In[1]:


from ML_utils import ML_utils
from ML_utils import ML_NLP_utlis
import shutil
import os

#從Dateset合併資料
data_pos_dir_path = r'D:\Backup\ml_data\Data_set\NLP\sentiment_analysis\utf-8\6000\6000\pos'
data_neg_dir_path = r'D:\Backup\ml_data\Data_set\NLP\sentiment_analysis\utf-8\6000\6000\neg'
data_pos_list = ML_utils.Merge_dir_file(data_pos_dir_path,save_name='merge_pos.txt',add_line_Feed=True,file_remove_LR=True)
data_neg_list = ML_utils.Merge_dir_file(data_neg_dir_path,save_name='merge_neg.txt',add_line_Feed=True,file_remove_LR=True)
#移至專案
dir_path = r'D:\Backup\ml_data\GitHub\ML_Sentiment_analysis\Dataset'
shutil.move(os.path.join(data_pos_dir_path,'merge_pos.txt'), os.path.join(dir_path,'merge_pos.txt'))
shutil.move(os.path.join(data_neg_dir_path,'merge_neg.txt'), os.path.join(dir_path,'merge_neg.txt'))
#前處理
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Remove_file_repeat_row, replace_old=True)
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Opencc_file, replace_old=True)
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Jieba_file_segmentation, replace_old=True)
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Filter_file_wlen,replace_old=False, regular=True,filter_='^merge',
                                max_word_num=60,file_head_name='cut_')
ML_utils.CallF_DirFile_save(dir_path, ML_utils.Trim_file_rows,replace_old=True,
                                filter_='cut_',row_num=1000)
sentiment_2_daat = ML_utils.Merge_dir_file(dir_path,save_name='sentiment_2_daat.txt',
                                           add_line_Feed=False,file_remove_LR=False,filter_='cut_')
with open(os.path.join(dir_path,'README.md'),'w',encoding='utf-8') as f:
    f.write(
    """### sentiment_2_daat ###\n格式為負評1000筆,正評1000筆共2000筆，以'\\n'隔開，每筆資料格式為長度小於等於60的字,以' '隔開。
    """)


# In[3]:


dir_path = r'D:\ml_data\GitHub\ML_Sentiment_analysis\word2vec_model'
ML_utils.Word2vec_train(None,None,dir_path=dir_path,save_name='word2vec_model')

