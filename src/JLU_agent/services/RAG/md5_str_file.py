from JLU_agent.config import chroma_config as config
import hashlib


"""提供md5相关操作的方法类，在该项目中md5用于上传知识文件的去重"""


class Md5Service:

    def check_md5(self, md5_str: str):
        """检查传入的md5字符串是否已经被处理
            return
                False: 文件未处理
                True: 已有记录
        """
        if not config.MD5_PATH.exists():
            #文件不存在,创建文件
            config.MD5_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.MD5_PATH.touch()
            return False
        else:
            with config.MD5_PATH.open('r', encoding="utf-8") as f:
                for line in f:
                    line = line.strip()     #处理字符串前后的空格回车
                    if line == md5_str:     #识别到记录
                        return True
            return False

    def save_md5(self, md5_str: str):
        """将传入的md5字符串记录到文件内保存"""
        config.MD5_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.MD5_PATH.open('a', encoding="utf-8") as f:
            f.write(md5_str + '\n')

    def get_string_md5(self, input_str: str, encoding='utf-8'):
        """将传入的字符串转换成md5字符串"""

        #将字符串转换成bytes字节数组
        input_str_bytes = input_str.encode(encoding=encoding)

        md5_object = hashlib.md5()              #创建md5对象
        md5_object.update(input_str_bytes)      #更新该对象的内容

        return md5_object.hexdigest()           #返回md5十六进制字符串