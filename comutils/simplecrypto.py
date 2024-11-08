from abc import abstractclassmethod
from typing import Dict
from Crypto.Cipher import AES  
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

BLOCK_SIZE = AES.block_size

# 至少填充BLOCK_SIZE个字节，方便后续清除
pad = lambda b: b + (BLOCK_SIZE - len(b) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(b) % BLOCK_SIZE).encode()

# 去掉填充数据
def unpad(cont:bytes):
    pad_char = cont[-1]
    cont = cont[0:-pad_char]
    return cont

path_key = 'key.txt'
def gen_key():
    import os
    key = os.urandom(32)
    with open(path_key, 'wb') as f:
        f.write(key)

'''
加密
'''
class Cryto(object):
    def __init__(self) -> None:
        self.key = Cryto.read_key()
        self.iv = self.key[0:16]  # 偏移量
        self.cipher = AES.new(key=self.key, mode=AES.MODE_CBC, IV=self.iv)
    @classmethod
    def read_buffer(self, path_file:Path):
        with open(path_file, 'rb') as f:
            buffer = f.read()
            return buffer
    @classmethod
    def read_key(cls)->bytes:
        key = '4r3oijfwefwefy7894yt2378293r&&223q=-sfwefsfwwe009234utpjpjojopj'
        byte_key = key.encode()[0:32]
        return byte_key
        # with open(path_key, 'rb') as f:
        #     return f.read()
    @abstractclassmethod
    def encrypt(self):
        raise NotImplemented

class Encryption(Cryto):
    def __init__(self, content) -> None:
        super().__init__()
        if isinstance(content, str):
            self.buffer = content.encode()
        elif isinstance(content, Path):
            self.buffer = Cryto.read_buffer(content)
        elif isinstance(content, bytes):
            self.buffer = content
        else:
            raise ('Error type')
    def __call__(self) ->bytes:
        return self.cipher.encrypt(pad(self.buffer))
    def encrypt(self):
        return self.__call__()

# 对文本进行加密
class TextEncryption(Encryption):
    def __init__(self, plain_text:str) -> None:
        super().__init__(plain_text)

# 对单个文件进行加密
class FileEncryption(Encryption):
    def __init__(self, path_file:str) -> None:
        super().__init__(Path(path_file))

'''
对多个字典buffer加密
参数为：
{
    key1:bytes1,
    key2:bytes2,
    ...
}

格式内容形式为：
n              |
key1 size1     |
key2 size2     | 为头部信息
...            |
keyn sizen     |
content1       |
content2       |
...            | 为内容本身
contentn       |
'''
class DictBufferEncryption(Encryption):
    def __init__(self, dict_buffer:Dict) -> None:
        self.dict_buffer = dict_buffer
        super().__init__(self.get_format_content())
    def get_format_content(self):
        head = bytes()
        content = bytes()
        head = head.__add__((str(len(self.dict_buffer)) + '\n').encode())
        for key, buffer in self.dict_buffer.items():
            if key.find('pytorch') != -1:    # key不能为'pytorch'
                continue
            else:
                head = head.__add__((key + ' ' + str(len(buffer)) + '\n').encode())
                content = content.__add__(buffer)
        content = head.__add__(content)
        return content
    def add_buffer(self, dict_buffer):
        for key, buffer in dict_buffer.items():
            if isinstance(buffer, str):
                buffer = buffer.encode()
            self.dict_buffer[key] = buffer
        self.buffer = self.get_format_content()    #  重新更新buffer

'''
处理多个文件加密（将多个文件加密为一个文件）
参数为：
{
    key1:path1,
    key2:path2
    ...
}
字典形式
'''
class MultiFileEncryption(DictBufferEncryption):
    def __init__(self, dict_path:Dict) -> None:
        self.dict_path = dict_path
        super().__init__(self.get_filelist_content())
    def get_filelist_content(self):
        content = {}
        for key, path in self.dict_path.items():
            content[key] = self.read_buffer(Path(path))
        return content

'''
字节解密
'''
class Decryption(Cryto):
    def __init__(self, content:bytes) -> None:
        super().__init__()
        self.content = content
    def __call__(self) ->bytes:
        cont = self.cipher.decrypt(self.content)
        return unpad(cont)
    def decrypt(self):
        return self.__call__()

'''
文件解密
'''
class FileDecryption(Decryption):
    def __init__(self, file_path:str) -> None:
        super().__init__(self.read_buffer(Path(file_path)))

if __name__ == '__main__':
    de = FileDecryption(r"D:\HCAI\Result\Project\Prj002_检测_子弹壳\train.enc")
    s = de().decode()
    with open(r"D:\HCAI\Result\Project\Prj002_检测_子弹壳\train.ini", encoding='utf8', mode='w') as f:
        f.write(s)