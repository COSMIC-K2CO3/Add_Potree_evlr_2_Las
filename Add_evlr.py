import laspy
from laspy.header import Version
from laspy.vlrs.vlrlist import VLRList, VLR

def add_potree(filepath, i):
    # 读取Potree文件并赋予evlr编号
    file = open(filepath, "rb")
    evlr_data = file.read()
    user_id = "hierarchy"
    record_id = i
    description = "the " + filepath + " of potree"
    evlr = VLR(user_id, record_id, description, evlr_data)
    file.close()
    return evlr

def get_filepath():
    # 获取文件路径
    filepath0 = "./LAS/FTZBZSB-JYZZZ-090.las"
    filepath1 = "./PoTree/hierarchy.bin"
    filepath2 = "./PoTree/octree.bin"
    filepath3 = "./PoTree/metadata.json"
    return filepath0, filepath1, filepath2, filepath3

def attach_evlr(filepath1, filepath2, filepath3):
    # 添加evlr
    evlr_1 = add_potree(filepath1, 1)
    evlr_2 = add_potree(filepath2, 2)
    evlr_3 = add_potree(filepath3, 3)
    evlrs = VLRList()
    evlrs.append(evlr_1)
    evlrs.append(evlr_2)
    evlrs.append(evlr_3)
    return evlrs

def main():
    # 主函数
    filepath0, filepath1, filepath2, filepath3 = get_filepath()

    # 读取原始las文件并转化版本
    las = laspy.read(filepath0)
    header = las.header
    # header.version = Version(1, 4)  # 1.4版本以上的las文件才支持扩展变长记录evlr

    # 添加evlr
    evlrs = attach_evlr(filepath1, filepath2, filepath3)

    # 写入新的las文件
    las.evlrs = evlrs
    las.header.evlrs = evlrs
    las.write("./Output/output_with_evlr.las")

main()