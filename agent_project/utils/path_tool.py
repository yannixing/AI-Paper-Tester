"""
为整个工程提供统一的绝对路径
"""
import os


def  get_project_root()->str:
    """获取工程的根目录"""
    current_file=os.path.abspath(__file__)
    current_dir=os.path.dirname(current_file)
    project_root=os.path.dirname(current_dir)
    return project_root

def get_abs_path(relative_path:str)->str:
    project_root=get_project_root()
    return os.path.join(project_root,relative_path)

if __name__ == '__main__':
    print("项目根:", get_project_root())
    print("chroma_db 路径:", get_abs_path("chroma_db"))