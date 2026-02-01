import sys
import os

def resource_path(relative_path):
    """获取资源的绝对路径，用于PyInstaller打包"""
    try:
        # PyInstaller创建的临时文件夹
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)