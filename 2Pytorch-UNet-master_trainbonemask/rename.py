import os
import re
import pdb
path = './tmp/'

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
regex = re.compile(r'\d\d+')

for file in fileList:
    # 设置旧文件名（就是路径+文件名）
    oldname = file  # os.sep添加系统分隔符

    replace = regex.findall(file)

    str = "".join(replace)

    # 设置新文件名
    newname = str + '_mask' + '.png'

    if not os.path.exists(newname):
        os.rename(path + oldname, path + newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

