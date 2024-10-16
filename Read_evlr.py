import laspy

def read_evlr():

    # 读取包含 EVLR 的 .las 文件
    las = laspy.read('./Output/output_with_evlr.las')

    # 检查文件是否包含 EVLR
    if las.evlrs:
        # 遍历所有的 EVLR 记录
        for idx, evlr in enumerate(las.evlrs):
            print(f"EVLR {idx + 1}:")
            print(f"  User ID: {evlr.user_id}")
            print(f"  Record ID: {evlr.record_id}")
            print(f"  Description: {evlr.description}")
            # print(f"  Data: {evlr.record_data}")
    else:
        print("No EVLRs found in the file.")