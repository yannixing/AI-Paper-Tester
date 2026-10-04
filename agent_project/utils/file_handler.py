import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from utils.logger_handler import logger

def get_file_md5_hex(filepath:str):
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")

    md5_obj=hashlib.md5()
    chunk_size=4096 #4KB 分片
    try:
        with open(filepath, 'rb') as f:
            while chunk:=f.read(chunk_size):
                md5_obj.update(chunk)
            md5_hex=md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")

def listdir_with_allowed_type(path:str,allowed_type:tuple[str]):
    files=[]
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
    for f in os.listdir(path):
        if f.endswith(allowed_type):
            files.append(os.path.join(path,f))

    return tuple(files)

def pdf_loder(filepath:str,passwd=None)->list[Document]:
    return PyPDFLoader(filepath,passwd).load()

def txt_loder(filepath:str,passwd=None)->list[Document]:
    return TextLoader(filepath,encoding="utf-8").load()

