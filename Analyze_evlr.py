import json
import laspy
import struct
import numpy as np

def read_evlr(filepath):
	# 打开包含 EVLR 的 LAS 文件
	las_file = laspy.read(filepath)

	# 提取 EVLR 数据，按照 record_id 区分 Potree 的三个文件
	metadata_json = None
	hierarchy_bin = None
	octree_bin = None

	for evlr in las_file.evlrs:
		if evlr.record_id == 1:
			hierarchy_bin = evlr.record_data  # hierarchy.bin
		elif evlr.record_id == 2:
			octree_bin = evlr.record_data  # octree.bin
		elif evlr.record_id == 3:
			metadata_json = evlr.record_data.decode('utf-8')  # metadata.json

	# 验证是否成功提取
	assert metadata_json is not None, "metadata.json 读取失败"
	assert hierarchy_bin is not None, "hierarchy.bin 读取失败"
	assert octree_bin is not None, "octree.bin 读取失败"

	return metadata_json, hierarchy_bin, octree_bin

def parse_metadata(metadata_json):
    metadata = json.loads(metadata_json)
    bounding_box = metadata['boundingBox']  # 获取全局空间范围
    scale = metadata['scale']  # 点云的尺度
    offset = metadata['offset']  # 原点偏移
    hierarchy_info = metadata['hierarchy']  # 获取层次信息
    first_chunk_size = hierarchy_info['firstChunkSize']
    step_size = hierarchy_info['stepSize']
    depth = hierarchy_info['depth']
    
    return bounding_box, scale, offset, first_chunk_size, step_size, depth

class OctreeNode:
    def __init__(self, offset, point_count):
        self.offset = offset  # 在 octree.bin 中的偏移
        self.point_count = point_count  # 节点中点的数量
        self.children = [None] * 8  # 初始化为 8 个子节点

def parse_hierarchy(hierarchy_bin, offset, depth, step_size):
    node_offset, point_count = struct.unpack_from('<qI', hierarchy_bin, offset)
    node = OctreeNode(node_offset, point_count)
    offset += 12  # 每个节点的数据占 12 字节（8 字节的偏移量和 4 字节的点数）

    if depth > 0:
        for i in range(8):  # 八叉树有 8 个子节点
            child, offset = parse_hierarchy(hierarchy_bin, offset, depth - 1, step_size)
            node.children[i] = child

    return node, offset

def compute_child_bounds(parent_bounds, child_index, offset, scale):
    # 根据子节点索引（0-7）和父节点边界划分子节点边界
    center = [(min_b + max_b) / 2 for min_b, max_b in zip(parent_bounds['min'], parent_bounds['max'])]
    
    child_min = list(parent_bounds['min'])
    child_max = list(parent_bounds['max'])

    if child_index & 1:
        child_min[0] = center[0]
    else:
        child_max[0] = center[0]

    if child_index & 2:
        child_min[1] = center[1]
    else:
        child_max[1] = center[1]

    if child_index & 4:
        child_min[2] = center[2]
    else:
        child_max[2] = center[2]

    # 根据 metadata.json 中的 offset 和 scale 来修正边界
    child_min = [c_min * s + o for c_min, s, o in zip(child_min, scale, offset)]
    child_max = [c_max * s + o for c_max, s, o in zip(child_max, scale, offset)]

    return {'min': child_min, 'max': child_max}

def is_node_in_bounds(node_bounds, query_bounds):
    # 判断节点空间与查询的 BoundingBox 是否相交
    return not (node_bounds['min'][0] > query_bounds['max'][0] or node_bounds['max'][0] < query_bounds['min'][0] or
                node_bounds['min'][1] > query_bounds['max'][1] or node_bounds['max'][1] < query_bounds['min'][1] or
                node_bounds['min'][2] > query_bounds['max'][2] or node_bounds['max'][2] < query_bounds['min'][2])

def load_point_data(node, octree_bin):
    points = []
    for i in range(node.point_count):
        point_offset = node.offset + i * 16  # 假设每个点占 16 字节（xyz + 其他属性）
        point = struct.unpack_from('<fff', octree_bin, point_offset)  # 假设每个点有3个float类型的坐标
        points.append(point)
    return points

def traverse_and_load_points(node, node_bounds, query_bounds, octree_bin, points, scale, offset):
    if not is_node_in_bounds(node_bounds, query_bounds):
        return

    # 加载当前节点的数据
    points.extend(load_point_data(node, octree_bin))

    # 递归遍历子节点
    for child_index, child in enumerate(node.children):
        if child is not None:
            child_bounds = compute_child_bounds(node_bounds, child_index, offset, scale)
            traverse_and_load_points(child, child_bounds, query_bounds, octree_bin, points, scale, offset)

def optimize_las_read(filepath, query_bounds):
    # 读取并解析 EVLR 数据
    metadata_json, hierarchy_bin, octree_bin = read_evlr(filepath)

    # 解析 metadata.json
    bounding_box, scale, offset, first_chunk_size, step_size, depth = parse_metadata(metadata_json)
    #query_bounds = bounding_box

    # 解析 hierarchy.bin，构建八叉树
    root_node, _ = parse_hierarchy(hierarchy_bin, 0, depth, step_size)

    # 准备存储点数据
    points = []

    # 递归选择八叉树节点，并加载数据
    traverse_and_load_points(root_node, bounding_box, query_bounds, octree_bin, points, scale, offset)

    return points

def main():
	filepath = "./Output/output_with_evlr.las"
	query_bounds = {'min': [-100, -100, -100], 'max': [100, 100, 100]}  # 示例查询范围
	points = optimize_las_read(filepath, query_bounds)
	print(f"Number of points in the query bounds: {len(points)}")

main()