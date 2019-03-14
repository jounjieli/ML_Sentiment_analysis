
# coding: utf-8

# In[30]:


"""
opencc: https://github.com/yichen0831/opencc-python
batch using dir_file_call_function()
"""
import os
import re
import sys 
import shutil

#取得文件資料列表(包括資料夾)
def get_dir_file_list(dir_path,filter_=None,distinguish=False,regular=False):
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

def str_remove_LR(str_in):
    str_out = re.sub(r"\n", r"", str_in)
    return str_out
def merge_dir_file(dir_path,save_name='dir_file_merge',filter=None,add_line_Feed=False,file_remove_LR=False):
    """
    ::parameter::
    filter: Filter files whose file names do not include strings
    file_remove_LR: read file and  remove LR
    add_line_Feed: add LR after file merge    
    """
    file_list = get_dir_file_list(dir_path,filter)
    file_merge = ""
    save_path = os.path.join(dir_path,save_name)
    for file_name in file_list: 
        file_path = os.path.join(dir_path,file_name)
        with open(file_path,'r',encoding='utf-8') as f:
            read_file = f.read()
            if file_remove_LR == True:
                read_file = str_remove_LR(read_file)
            file_merge += read_file
            if add_line_Feed == True:
                file_merge = file_merge + '\n'
    with open(save_path,'w',encoding='utf-8') as save:
        save.write(file_merge)
    return file_merge

def file_remove_repeat_row(file_path,save_path,replace_old=False):
    """
    batch using dir_file_call_function()
    """
    with open(file_path, 'r',encoding='utf-8') as f:
        out = ''.join(list(set([i for i in f])))
    with open(save_path, 'w',encoding='utf-8') as f:
        f.write(out)
    if replace_old == True:
        shutil.move(save_path,file_path)
    
def dir_file_call_function(dir_path,function,file_head_name='new_',replace_old=False,filter_=None,regular=False,**kw):
    """
    **kw ==>> function(file_path=file_path,save_path=save_path,**kw)    ex. conversion='s2t'
    """
    file_list = get_dir_file_list(dir_path=dir_path,filter_=filter_,regular=regular,distinguish=True)
    file_list = file_list[0]
    for file_name in file_list:
        file_path = os.path.join(dir_path,file_name)
        save_path = os.path.join(dir_path,file_head_name+file_name)
        function(file_path=file_path,save_path=save_path,replace_old=replace_old,**kw)

