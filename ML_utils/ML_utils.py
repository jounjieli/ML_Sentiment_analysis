
# coding: utf-8

# In[25]:


"""
opencc: https://github.com/yichen0831/opencc-python
batch using dir_file_call_function()
"""
import os
import re
import sys 
import shutil
import logging

#取得文件資料列表(包括資料夾)
def Get_dir_file_list(dir_path,filter_=None,distinguish=False,regular=False):
    """
    filter: Filter files whose file names do not include strings
    distinguish: Distinguish between folders and file,True=Distinguish
    """
    dir_path = dir_path
    file_list = os.listdir(dir_path)
    file_list_ = []
    dir_list_ = []
    #---#
    if distinguish == False:
        if filter_ == None:
            for file in file_list:
                file_list_.append(file)
        else:
            if regular == False:
                for file in file_list:
                    if filter_ in file:
                        file_list_.append(file)
            if regular == True:
                for file in file_list:
                    if re.search(filter_, file) != None:
                        file_list_.append(file)       
        file_list = file_list_
    #---#
    if distinguish == True:
        if filter_ == None:
            for file in file_list:
                if os.path.isfile( os.path.join(dir_path,file) ) == True:
                    file_list_.append(file)
                elif os.path.isdir( os.path.join(dir_path,file) ) == True:
                    dir_list_.append(file)
        else:
            if regular == False:
                for file in file_list:
                    if filter_ in file:
                        if os.path.isfile( os.path.join(dir_path,file) ) == True:
                            file_list_.append(file)
                        elif os.path.isdir( os.path.join(dir_path,file) ) == True:
                            dir_list_.append(file)
            if regular == True:
                for file in file_list:
                    if re.search(filter_, file) != None:
                        if os.path.isfile( os.path.join(dir_path,file) ) == True:
                            file_list_.append(file)
                        elif os.path.isdir( os.path.join(dir_path,file) ) == True:
                            dir_list_.append(file)
        file_list = [file_list_,dir_list_]
    return file_list

def CallF_DirFile_save(dir_path,function,file_head_name='new_',replace_old=False,filter_=None,regular=False,**kw):
    """
    **kw ==>> function(file_path=file_path,save_path=save_path,**kw)    ex. conversion='s2t'
    expansion: def function(file_path,save_path,replace_old,....)
    """
    file_list = Get_dir_file_list(dir_path=dir_path,filter_=filter_,regular=regular,distinguish=True)
    file_list = file_list[0]
    for file_name in file_list:
        file_path = os.path.join(dir_path,file_name)
        save_path = os.path.join(dir_path,file_head_name+file_name)
        function(file_path=file_path,save_path=save_path,replace_old=replace_old,**kw)
        
def CallF_DirFile(dir_path,function,filter_=None,regular=False,**kw):
    """
    **kw ==>> function(file_path=file_path,save_path=save_path,**kw)    ex. conversion='s2t'
    expansion: def function(file_path,save_path,replace_old,....)
    """
    file_list = Get_dir_file_list(dir_path=dir_path,filter_=filter_,regular=regular,distinguish=True)
    file_list = file_list[0]
    save_list = []
    for file_name in file_list:
        file_path = os.path.join(dir_path,file_name)
        save_list.append(function(file_path=file_path,**kw))
    return save_list

def Merge_dir_file(dir_path,save_name='dir_file_merge',filter_=None,regular=False,
                   add_line_Feed=True,file_remove_LR=False,encoding='utf-8'):
    """
    ::parameter::
    filter: Filter files whose file names do not include strings
    file_remove_LR: read file and  remove LR
    add_line_Feed: add LR after file merge    
    """
    file_list = Get_dir_file_list(dir_path,filter_=filter_,regular=False)
    file_merge = ""
    save_path = os.path.join(dir_path,save_name)
    for file_name in file_list: 
        file_path = os.path.join(dir_path,file_name)
        with open(file_path,'r',encoding=encoding) as f:
            read_file = f.read()
            if file_remove_LR == True:
                read_file = Remove_str_LR(read_file)
            file_merge += read_file
            if add_line_Feed == True:
                file_merge = file_merge + '\n'
    with open(save_path,'w',encoding=encoding) as save:
        save.write(file_merge)
    return file_merge
def Remove_str_LR(str_in):
    str_out = re.sub(r"\n", r"", str_in)
    return str_out
                        
def Remove_file_repeat_row(file_path,save_path,replace_old=False,encoding='utf-8'):
    """
    batch using dir_file_call_function()
    """
    with open(file_path, 'r',encoding=encoding) as f:
        out = ''.join(list(set([i for i in f])))
    with open(save_path, 'w',encoding=encoding) as f:
        f.write(out)
    if replace_old == True:
        shutil.move(save_path,file_path)

def WordToList_file(file_path,encoding='utf-8',max_word_num=None,word_split=' ',row_split='\n',padding=None):
    with open(file_path,'r',encoding=encoding) as f:
        save_list = []
        if max_word_num == None:
            for line in f:
                line_ = line.strip('\n').strip()
                word_list = line_.split(word_split)
                save_list.append(word_list)
        if max_word_num != None:
            if padding == None:
                for line in f:
                    line_ = line.strip('\n').strip()
                    word_list = line_.split(word_split)
                    if len(word_list) <= max_word_num:
                        save_list.append(word_list)
            if padding != None:
                for line in f:
                    padding_seq = ''
                    line_ = line.strip('\n').strip()
                    word_list = line_.split(word_split)
                    if len(word_list) <= max_word_num:
                        for num in range(max_word_num-len(word_list)):
                            word_list.append(padding)
                        save_list.append(word_list)
    return save_list
        
def Filter_file_wlen(file_path,save_path,max_word_num,replace_old=False,encoding='utf-8',
                     word_split=' ',row_split='\n',padding=None):
    """
    Filter out longer of line than max_word_num
    """
    with open(file_path,'r',encoding=encoding) as f:
        with open(save_path, 'w',encoding=encoding) as f_wrtie:
            if padding == None:
                for line in f:
                    line_ = line.strip('\n')
                    word_list = line_.split(word_split)
                    if len(word_list) <= max_word_num:
                        line_ = line_+ row_split
                        f_wrtie.write(line_)
            if padding != None:
                for line in f:
                    padding_seq = ''
                    line_ = line.strip('\n').strip()
                    word_list = line_.split(word_split)
                    if len(word_list) <= max_word_num:
                        for num in range(max_word_num-len(word_list)):
                            word_list.append(padding)
                        for seq in word_list:
                            padding_seq = padding_seq+ seq + word_split
                        padding_seq = padding_seq + row_split
                        f_wrtie.write(padding_seq)
    if replace_old == True:
        shutil.move(save_path,file_path)
        
def Padding_file_lword(file_path,save_path,max_word_num,replace_old=False,encoding='utf-8',
                       padding='my_padding_str',word_split=' ',row_split='\n'):
    """
    The number of words in the line does not exceed max_word_num, run padding.
    """
    with open(file_path,'r',encoding=encoding) as f:
        with open(save_path, 'w',encoding=encoding) as f_wrtie:
            for line in f:
                #依行讀取
                padding_seq = ''
                line_ = line.strip('\n').strip()
                word_list = line_.split(word_split)
                if len(word_list) <= max_word_num:
                    for num in range(max_word_num-len(word_list)):
                        word_list.append(padding)
                    for seq in word_list:
                        padding_seq = padding_seq+ seq + word_split
                    padding_seq = padding_seq + row_split
                    f_wrtie.write(padding_seq)
    if replace_old == True:
        shutil.move(save_path,file_path)    

def Trim_file_rows(file_path,save_path,row_num=1000,n_times=False,replace_old=False,encoding='utf-8',sep='\n'):
    import pandas as pd
    num = 1
    data = pd.read_csv(file_path,encoding=encoding,sep=sep,header=-1)
    if n_times != False:
        num = len(data)//row_num
    data[:num*row_num].to_csv(save_path,sep='\n',header=False,index=False)
    if replace_old == True:
        shutil.move(save_path,file_path)

def Jieba_str_segmentation(string,delimiter=' ',stopword_path=None,split=False,encoding='utf-8'):
    """
    split: string to list
    stopword_path: delimiter of stopword must is '\n'
    """
    import jieba
    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')
    # load stopwords set
    #將停用詞每row分別加進集合
    stopword_set = set()
    if stopword_path != None:
        #設置停用詞讀取路徑
        with open(stopword_path,'r', encoding=encoding) as stopwords:
            for stopword in stopwords:
                stopword_set.add(stopword.strip('\n'))   #移除頭尾換行 strip('\n')
    output = ''
    string = string.strip('\n')
    words = jieba.cut(string, cut_all=False,HMM=True)    #進行斷詞
    for word in words:
        #依每個詞判斷是否為停用詞(不是就寫入)
        if word not in stopword_set:
            output = output+word+delimiter
    if split == True:
        output = output.split(delimiter)
    return output

def Opencc_str(string,conversion='s2t'):
    """
    opencc:
    https://github.com/yichen0831/opencc-python
    conversion: 
    hk2s: Traditional Chinese (Hong Kong standard) to Simplified Chinese
    s2hk: Simplified Chinese to Traditional Chinese (Hong Kong standard)
    s2t: Simplified Chinese to Traditional Chinese
    s2tw: Simplified Chinese to Traditional Chinese (Taiwan standard)
    s2twp: Simplified Chinese to Traditional Chinese (Taiwan standard, with phrases)
    t2hk: Traditional Chinese to Traditional Chinese (Hong Kong standard)
    t2s: Traditional Chinese to Simplified Chinese
    t2tw: Traditional Chinese to Traditional Chinese (Taiwan standard)
    tw2s: Traditional Chinese (Taiwan standard) to Simplified Chinese
    tw2sp: Traditional Chinese (Taiwan standard) to Simplified Chinese (with phrases)
    """
    from opencc import OpenCC
    cc = OpenCC(conversion)
    out =  cc.convert(string)
    return out

def Jieba_file_segmentation(file_path,save_path,replace_old=False,word_delimiter=' ',
                            file_delimiter='\n',stopword_path=None,encoding='utf-8'):
    """
    batch using dir_file_call_function()
    close log using "logging.disable(lvl)"
    https://docs.python.org/3/library/logging.html
    stopword_path: delimiter of stopword must is '\n'
    """
    import jieba
    #設置log格式，以及print的log等級
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')
    #將停用詞每row分別加進集合
    stopword_set = set()
    if stopword_path != None:
        with open(stopword_path,'r', encoding=encoding) as stopwords:
            for stopword in stopwords:
                stopword_set.add(stopword.strip('\n'))   #移除頭尾換行 strip('\n')
    #open write file
    output = open(save_path, 'w', encoding=encoding)
    with open(file_path, 'r', encoding=encoding) as content :
        #每一行都切成一個iter
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False,HMM=True)    #進行斷詞
            for word in words:
                #依每個詞判斷是否為停用詞(不是就寫入)
                if word not in stopword_set:
                    output.write(word + word_delimiter)     #每一行的iter(詞)以空格隔開
            output.write(file_delimiter)      #iter完以換行符區隔
            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
    output.close()
    if replace_old == True:
        shutil.move(save_path,file_path)

def Opencc_file(file_path,save_path,replace_old=False,conversion='s2t',encoding='utf-8'):
    """
    batch using dir_file_call_function()
    close log using "logging.disable(lvl)"
    https://docs.python.org/3/library/logging.html
    opencc:
    https://github.com/yichen0831/opencc-python
    conversion: 
    hk2s: Traditional Chinese (Hong Kong standard) to Simplified Chinese
    s2hk: Simplified Chinese to Traditional Chinese (Hong Kong standard)
    s2t: Simplified Chinese to Traditional Chinese
    s2tw: Simplified Chinese to Traditional Chinese (Taiwan standard)
    s2twp: Simplified Chinese to Traditional Chinese (Taiwan standard, with phrases)
    t2hk: Traditional Chinese to Traditional Chinese (Hong Kong standard)
    t2s: Traditional Chinese to Simplified Chinese
    t2tw: Traditional Chinese to Traditional Chinese (Taiwan standard)
    tw2s: Traditional Chinese (Taiwan standard) to Simplified Chinese
    tw2sp: Traditional Chinese (Taiwan standard) to Simplified Chinese (with phrases)
    """
    from opencc import OpenCC
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    cc = OpenCC(conversion)
    string = ''
    with open(file_path,'r',encoding=encoding) as read_f:
        with open(save_path,'w',encoding=encoding) as write_f:
            for texts_num,read_line in enumerate(read_f):
                file_str =  cc.convert(read_line)
                write_f.writelines(file_str)
                if (texts_num + 1) % 10000 == 0:
                    logging.info("已完成前 %d 行的轉換" % (texts_num + 1))
    if replace_old == True:
        shutil.move(save_path,file_path)
        
def Word2vec_train(file_path,save_path,dir_path=None,save_name='word2vec_model',replace_old=False,
                   model_size=300,model_window=10,model_min_count=5,**kw):
    """
    batch train usage: set dir_path、save_name, file_path = None, save_path = None
    if Multiple files using dir_path
    """
    from gensim.models import word2vec
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # https://radimrehurek.com/gensim/models/word2vec.html
    if file_path != None:
        #單檔案
        sentences = word2vec.LineSentence(file_path)
        model = word2vec.Word2Vec(sentences, size=model_size,window=model_window,min_count=model_min_count,**kw)
        #保存模型，供日後使用
        model.save(save_path)
    if dir_path != None and file_path == None:
        #多檔案
        sentences = word2vec.PathLineSentences(dir_path)
        model = word2vec.Word2Vec(sentences, size=model_size,window=model_window,min_count=model_min_count,**kw)
        #保存模型，供日後使用
        model.save(os.path.join(dir_path,save_name))
    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name") 

