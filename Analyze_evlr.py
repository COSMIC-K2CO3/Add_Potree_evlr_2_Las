import os 
import json
import laspy
import struct
import numpy as np
from enum import Enum
from typing import Callable
from typing import List, Union

class Attribute:
    def __init__(self, name: str = "", description: str = "", size: int = 0,
                 num_elements: int = 0, element_size: int = 0, attribute_type=None):
        self.name = name
        self.description = description
        self.size = size
        self.num_elements = num_elements
        self.element_size = element_size
        self.type = {"int8":np.int8,"int16":np.int16,"int32":np.int32,"int64":np.int64,"uint8":np.uint8,"uint16":np.uint16,
                     "uint32":np.uint32,"uint64":np.uint64,"float":np.float32,"double":np.float64,}.get(attribute_type)
        self.min = np.array([ np.inf,  np.inf,  np.inf])
        self.max = np.array([-np.inf, -np.inf, -np.inf])
        
    def __repr__(self):
        return str(self.__dict__)
                
class Attributes:
    def __init__(self, attributes: List[Attribute] = None):
        if attributes is None:
            attributes = []
        self.list = attributes
        self.bytes = sum([attribute.size for attribute in attributes])
        self.pos_scale = np.array([1.0, 1.0, 1.0])
        self.pos_offset = np.array([0.0, 0.0, 0.0])

    def add(self, attribute: Attribute):
        self.list.append(attribute)
        self.bytes += attribute.size

    def get_offset(self, name: str) -> int:
        offset = 0
        for attribute in self.list:
            if attribute.name == name:
                return offset
            offset += attribute.size
        return -1

    def get(self, name: str) -> Union[Attribute, None]:
        for attribute in self.list:
            if attribute.name == name:
                return attribute
        return None
    
    def __repr__(self):
        return str(self.list)

class PotreePoints:
    def __init__(self,metadata_json):
        self.attributes = self.parse_attributes(metadata_json)
        self.attribute_buffers_map = {}
        self.num_points = 0

    def parse_attributes(self,metadata: dict) -> Attributes:
        attribute_list = []
        js_attributes = metadata["attributes"]

        for js_attribute in js_attributes:
            name = js_attribute["name"]
            description = js_attribute["description"]
            size = js_attribute["size"]
            num_elements = js_attribute["numElements"]
            element_size = js_attribute["elementSize"]
            attr_type = js_attribute["type"]

            js_min = np.asarray(list(map(float,js_attribute["min"])))
            js_max = np.asarray(list(map(float,js_attribute["max"])))

            # 创建属性对象
            attribute = Attribute(name=name, size=size, num_elements=num_elements, element_size=element_size, attribute_type=attr_type)

            # 设置属性的最小值和最大值，如果长度小于3，则赋值
            if len(js_min)<=3 and len(js_max)<=3:
                attribute.min[:len(js_min)] = js_min
                attribute.max[:len(js_max)] = js_max
            attribute_list.append(attribute)

        attributes = Attributes(attribute_list)
        attributes.pos_scale = np.array(metadata["scale"])
        attributes.pos_offset = np.array(metadata["offset"])

        return attributes

    def add_attribute_buffer(self, attribute: Attribute, buffer):
        self.attribute_buffers_map[attribute.name] = buffer

    def remove_attribute(self, attribute_name: str):
        index = -1
        for i, attribute in enumerate(self.attributes.list):
            if attribute.name == attribute_name:
                index = i
                break
        if index >= 0:
            del self.attributes
            del self.attribute_buffers_map[attribute_name]

    def get_raw_data(self,key):
        if key not in self.attribute_buffers_map.keys():
            raise ValueError(f'no {key} data!')
            # return None,None
        buffer = self.attribute_buffers_map[key]
        attr = self.attributes.get(key)
        # print(attr.name,attr.type,attr.num_elements)
        ps = np.frombuffer(buffer.data, dtype=attr.type, count=self.num_points*attr.num_elements)
        return attr,ps.reshape(-1,attr.num_elements)

    def get_position(self) -> np.ndarray:
        attr,position = self.get_raw_data('position')
        position = position * self.attributes.pos_scale + self.attributes.pos_offset
        return position
    
    def get_rgb(self) -> np.ndarray:
        attr,rgb = self.get_raw_data('rgb')
        return rgb /attr.max.max()
    
    def get_point_source_id(self) -> np.ndarray:
        attr,ids = self.get_raw_data('point source id')
        return ids
    
    def get_intensity(self) -> np.ndarray:
        attr,intensity = self.get_raw_data('intensity')
        return intensity /attr.max.max()
        
    def get_classification(self) -> np.ndarray:
        attr,classification = self.get_raw_data('classification')
        return classification
    
    def get_area(self):
        ps = self.get_position()
        return np.prod(ps.max(0)[:2] - ps.min(0)[:2])
    
    def get_volumn(self):
        ps = self.get_position()
        return np.prod(ps.max(0)[:3] - ps.min(0)[:3])    

class PotreeNode:    
    class NodeType(Enum):
        NORMAL = 0  # 普通节点
        LEAF = 1    # 叶子节点
        PROXY = 2   # 代理节点

    class AABB: # Axis Aligned Bounding Box, 轴对齐的边界框
        def __init__(self, min_coords, max_coords):
            self.min = np.asarray(min_coords)
            self.max = np.asarray(max_coords)
            
        def __str__(self):
            return str((self.min,self.max)) # 当用print输出时，输出的是这个
        
        def area(self):
            return np.prod(self.max[:2] - self.min[:2]) # 面积
        
        def volumn(self):
            return np.prod(self.max - self.min) # 体积

    def __init__(self, path='', name = '', aabb = None):
        self.name = name
        self.aabb = aabb
        self.parent = None
        self.children:list[PotreeNode] = [None] * 8
        self.node_type = -1
        self.byte_offset = 0
        self.byte_size = 0
        self.num_points = 0
        self.path = path
        
    def __repr__(self):
        return self.name#str(self.__dict__)

    def level(self):
        return len(self.name) - 1

    def get_all_children(self):
        nodes=[]
        self.traverse(lambda node: nodes.append(node))
        return nodes

    def read_all_children_nodes(self):
        nodes:list[PotreeNode]=self.get_all_children()
        pp = [n.read_node() for n in nodes]
        return pp

    def traverse(self, callback: Callable[['PotreeNode'], None]):   # 递归遍历树（深度优先）, Callable用于检查对象是否可调用, callback不返回值
        callback(self)
        for child in self.children:
            if child is not None:
                child.traverse(callback)

    def write_node(self,data_dict:dict):#{'classification':np.ones(n0.num_points,dtype=np.uint8)})
        tmpp = self
        while tmpp.path == '':
            tmpp = tmpp.parent
        potree_path = tmpp.path

        las_file = laspy.read(potree_path)

        attributes_json = None
        octree_bin = None

        for evlr in las_file.evlrs:
            if evlr.record_id == 3:
                attributes_json = json.loads(evlr.record_data.decode('utf-8'))  # metadata.json
            
        points = PotreePoints(attributes_json)
        points.num_points = self.num_points

        for evlr in las_file.evlrs:
            if evlr.record_id == 2:
                octree_bin = evlr.record_data  # octree.bin
                is_brotli_encoded = (attributes_json["encoding"] == "BROTLI")
                if is_brotli_encoded:
                    raise ValueError('brotli encoded is not support!')
                else:
                    attribute_offset = 0
                    for attribute in points.attributes.list:
                        attribute_data_size = attribute.size * self.num_points
                        offset_target = 0
                        if attribute.name in data_dict.keys():
                            data:np.ndarray = data_dict[attribute.name]
                            if len(data.flatten().tobytes())!=attribute_data_size:
                                raise ValueError(f'data size:{data.shape} is not in right bytes!(attribute.size:{attribute.size} , num_points:{self.num_points})')
                            for i in range(points.num_points):
                                base_offset = i * points.attributes.bytes + attribute_offset
                                octree_bin.seek(self.byte_offset + base_offset)
                                octree_bin.write(data[i].tobytes())
                                offset_target += attribute.size
                        attribute_offset += attribute.size
    
    def write_uniform_classification(self, lable=0):
        return self.write_node({'classification':np.ones(self.num_points,dtype=np.uint8)*lable})
                        
    def write_classification(self, data):
        return self.write_node({'classification':data})
    
    def write_uniform_rgb(self, color=(0,0,0)):# 0.0 ~ 1.0
        data = np.ones((self.num_points,3),dtype=np.uint16)
        data[:,0] = int(color[0] * np.iinfo(np.uint16).max)
        data[:,1] = int(color[1] * np.iinfo(np.uint16).max)
        data[:,2] = int(color[2] * np.iinfo(np.uint16).max)
        return self.write_node({'rgb':data})
    
    def write_rgb(self, data):
        return self.write_node({'rgb':data})

    def read_node(self):     
        tmpp = self
        while tmpp.path == '':
            tmpp = tmpp.parent
        potree_path = tmpp.path
        las_file = laspy.read(potree_path)
        for evlr in las_file.evlrs:
            if evlr.record_id == 3:
                attributes_json = json.loads(evlr.record_data.decode('utf-8'))
            
        points = PotreePoints(attributes_json)
        points.num_points = self.num_points

        for evlr in las_file.evlrs:
            if evlr.record_id == 2:
                #file = evlr.record_data
                #file.seek(self.byte_offset)
                #data = file.read(self.byte_size)
                data = evlr.record_data

        is_brotli_encoded = (attributes_json["encoding"] == "BROTLI")   # 检查编码格式
        if is_brotli_encoded:
            raise ValueError('brotli encoded is not support!')

        else:
            attribute_offset = 0
            for attribute in points.attributes.list:
                attribute_data_size = attribute.size * self.num_points
                buffer = np.empty(attribute_data_size, dtype=np.uint8) # 创建一个空的字节数组
                offset_target = 0
                for i in range(points.num_points):
                    base_offset = i * points.attributes.bytes + attribute_offset    # 计算当前点在octree_bin中的偏移量
                    raw = data[ base_offset : base_offset + attribute.size]         # 从data中读取当前点在octree_bin中的数据
                    buffer[offset_target:offset_target + attribute.size] = np.frombuffer(raw, dtype=np.uint8)   # 将读取的数据转换为np.uint8类型并存储到buffer中
                    offset_target += attribute.size

                points.add_attribute_buffer(attribute, buffer)  # 将buffer添加到points中
                attribute_offset += attribute.size

        return points

class Potree:
    def __init__(self, path=None):
        print(os.path)
        assert os.path.isfile(path) # 断言文件存在

        self.root = None
        self.nodes = []
        self.path = path
        if path is not None:
            self.load()
    
    def load_hierarchy_recursive(self, root: PotreeNode, data: bytes, offset: int, size: int):
        bytesPerNode = 22   # 每个节点的字节数
        numNodes = size // bytesPerNode

        nodes = [root]  # 初始化节点列表，包含根节点

        for i in range(numNodes):
            current = nodes[i]

            offsetNode = offset + i * bytesPerNode  # 计算当前节点的偏移量
            type, childMask, numPoints, byteOffset, byteSize = struct.unpack_from('<BBIqq', buffer=data, offset=offsetNode) # 格式:unsigned char, unsigned char, unsigned int, unsigned long long, unsigned long long, 可读的缓冲区字对象data(即hierarchy.bin)用于提取数据, 从何处开始读取数据

            current.byte_offset = byteOffset
            current.byte_size = byteSize
            current.num_points = numPoints
            current.node_type = type

            if current.node_type == PotreeNode.NodeType.PROXY.value:    # 如果节点类型为代理节点
                self.load_hierarchy_recursive(current, data, byteOffset, byteSize)  # 递归加载子节点
            else:
                for childIndex in range(8): # 遍历子节点，检查是否存在
                    childExists = ((1 << childIndex) & childMask) != 0

                    if not childExists:
                        continue

                    childName = current.name + str(childIndex)  # 若存在，则创建子节点对象，并设置属性

                    child = PotreeNode(name=childName, aabb=self.child_AABB(current.aabb, childIndex))
                    current.children[childIndex] = child
                    child.parent = current

                    nodes.append(child) # 将子节点添加到节点列表中

    def child_AABB(self, aabb:PotreeNode.AABB, index):
        min_coords,max_coords = aabb.min.copy(),aabb.max.copy()
        size = [max_coord - min_coord for max_coord, min_coord in zip(aabb.max, aabb.min)]
        min_coords[2] += ( size[2] / 2 if (index & 0b0001) > 0 else -(size[2] / 2) )
        min_coords[1] += ( size[1] / 2 if (index & 0b0010) > 0 else -(size[1] / 2) )
        min_coords[0] += ( size[0] / 2 if (index & 0b0100) > 0 else -(size[0] / 2) )
        return PotreeNode.AABB(min_coords, max_coords)

    def load(self, path=None):
        if self.path is not None:path = self.path        
        assert self.path is not None or path is not None

        # 打开包含 EVLR 的 LAS 文件
        las_file = laspy.read(path)

        # 提取 EVLR 数据，按照 record_id 区分 Potree 的三个文件
        metadata = None
        data = None

        for evlr in las_file.evlrs:
            if evlr.record_id == 1:
                data = evlr.record_data  # hierarchy.bin
            elif evlr.record_id == 3:
                metadata = json.loads(evlr.record_data.decode('utf-8'))  # metadata.json

        jsHierarchy = metadata["hierarchy"]
        firstChunkSize = jsHierarchy["firstChunkSize"]
        # stepSize = jsHierarchy["stepSize"]
        # depth = jsHierarchy["depth"]

        aabb = PotreeNode.AABB(metadata["boundingBox"]["min"],metadata["boundingBox"]["max"])
        self.root = PotreeNode(path, name="r", aabb=aabb)   # 初始化根节点
        self.load_hierarchy_recursive(self.root, data, offset = 0, size = firstChunkSize)
        self.nodes = []
        self.root.traverse(lambda node: self.nodes.append(node))    # 遍历所有节点，将节点添加到 nodes 列表中
        return self

    def bfs(self,node=[],depth=0,resdict={}):   # 广度优先搜索，节点列表，深度，每层字典
        node:list[PotreeNode] = list(filter(lambda x: x is not None, node)) # 过滤掉 None 值，确保节点列表中只包含非 None 值
        if len(node)==0:return  # 如果节点列表为空，则遍历完，返回
        res = []    # 列表用于存储下一层的节点
        resdict[depth] = node   # 将当前层的节点存储到字典中，键为深度
        for i in resdict[depth]:
            i:PotreeNode=i
            res += i.children   # 将当前层的节点的子节点添加到下一层节点列表中
        self.bfs(res,depth+1,resdict)   # 递归调用，遍历下一层节点
    
    def get_nodes_LOD_dict(self)->dict[int,PotreeNode]:
        res = {}
        self.bfs([self.root],0,res)
        return res
        
    def get_max_LOD(self):
        return max(self.get_nodes_LOD_dict().keys())
    
    def get_nodes_by_LOD(self, lod=0)->list[PotreeNode]:
        assert type(lod) == int, 'nodes key must be int!'
        return self.get_nodes_LOD_dict().get(lod,[])

    def get_point_size_by_LOD(self,lod=0):
        return sum([n.num_points for n in self.get_nodes_by_LOD(lod)])

    def get_data_by_LOD(self,data_name:list[str]=['position'],lod=0):        
        res = []
        nodes = self.get_nodes_by_LOD(lod)  # 获取节点列表
        print('read points :',sum([n.num_points for n in nodes]))    # 打印节点列表中所有节点的点数总和
        if len(nodes)==0:return res # 如果节点列表为空，则返回空列表

        nodes = [n.read_node() for n in nodes]  # 读取节点的数据，更新节点列表
        method_list = [func for func in dir(PotreePoints) if callable(getattr(PotreePoints, func))] # 获取 PotreePoints 类中所有可调用的方法列表
        for name in data_name:
            name = name.replace(' ', '_')   # 将空格替换为下划线
            if 'get_'+name in method_list:  # 检查方法列表中是否存在 'get_'+name 方法
                ps = [getattr(n, 'get_'+name)() for n in nodes] # 对节点调用 'get_'+name 方法并存储结果于 ps 列表中
                res.append(np.vstack(ps))   # 将ps列表中的所有数组垂直堆叠成一个数组，添加进结果列表中
            else:
                raise ('no function get_'+name)
        return res
    
    def get_position_by_LOD(self, lod=0):
        res = self.get_data_by_LOD(data_name=['position'],lod=lod)
        return res[0]
    
    def get_rgb_by_LOD(self, lod=0):
        res = self.get_data_by_LOD(data_name=['rgb'],lod=lod)
        return res[0]
    
    def get_intensity_by_LOD(self, lod=0):
        res = self.get_data_by_LOD(data_name=['intensity'],lod=lod)
        return res[0]
