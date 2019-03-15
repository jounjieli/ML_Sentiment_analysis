
# coding: utf-8

# In[1]:


import jieba
from opencc import OpenCC
from gensim.models import word2vec
import logging
import shutil
import os

def jieba_string_segmentation(string,delimiter=' ',stopword_path=None,split=False,encoding='utf-8'):
    """
    split: string to list
    stopword_path: delimiter of stopword must is '\n'
    """
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

def str_to_opencc(string,conversion='s2t'):
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
    cc = OpenCC(conversion)
    out =  cc.convert(string)
    return out

def jieba_file_segmentation(file_path,save_path,replace_old=False,word_delimiter=' ',
                            file_delimiter='\n',stopword_path=None,encoding='utf-8'):
    """
    batch using dir_file_call_function()
    close log using "logging.disable(lvl)"
    https://docs.python.org/3/library/logging.html
    stopword_path: delimiter of stopword must is '\n'
    """
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

def file_to_opencc(file_path,save_path,replace_old=False,conversion='s2t',encoding='utf-8'):
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
        
def word2vec_train(file_path,save_path,dir_path=None,save_name='word2vec_model',replace_old=False,
                   model_size=300,model_window=10,model_min_count=5,**kw):
    """
    batch train usage: set dir_path、save_name, file_path = None, save_path = None
    if Multiple files using dir_path
    """
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

