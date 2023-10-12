from functools import reduce
import numpy as np
import time
import os
from tqdm import tqdm
from pymilvus import (
    connections, list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
import torch
from torchvision import models, transforms
from img2features import img2feature
from PIL import Image
import time
import numpy as np
import random
from io import BytesIO
import base64
import os
import uuid


class MilvusManager:
    def __init__(self, field_name, host, port, dim):
        self.field_name = field_name
        self.host = host
        self.port = port
        self.dim = dim
        self.default_fields = [
            FieldSchema(name="milvus_id",
                        dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="image_name",
                        dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="img_data", dtype=DataType.VARCHAR,
                        max_length=65535),  # 添加这一行来存储图片的Base64编码
            FieldSchema(name="feature",
                        dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]

    def create_table(self):
        connections.connect(host=self.host, port=self.port)

        if utility.has_collection(self.field_name):
            old_collection = Collection(name=self.field_name)
            print('Collection already exists. Dropping...')
            old_collection.drop()

        default_schema = CollectionSchema(
            fields=self.default_fields, description="test collection")

        print(f"\nCreate collection...")
        collection = Collection(name=self.field_name, schema=default_schema)
        print(f"\nCreate index...")
        default_index = {"index_type": "FLAT", "params": {
            "nlist": 128}, "metric_type": "L2"}
        collection.create_index(field_name="feature",
                                index_params=default_index)
        print(f"\nCreate index...is OKOKOKOKOK")
        collection.load()

    def save_image_to_disk(self, img_data):
        # 从Base64编码生成一个图片对象
        img = self.decode_image(img_data)

        # 为图片生成一个唯一的文件名
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join('uploads', filename)

        # 保存图片到磁盘
        img.save(filepath)

        return filename

    def decode_image(self, img_data):
        base64_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        return Image.open(BytesIO(img_bytes))

    def insert_data(self, id, img_data, img_name):
        connections.connect(host=self.host, port=self.port)
        default_schema = CollectionSchema(
            fields=self.default_fields, description="test collection")
        collection = Collection(name=self.field_name, schema=default_schema)

        # Convert img_data (in bytes format) to an image
        img = self.decode_image(img_data)  # 使用此函数将Base64编码转换为图片
        vectors = img2feature(img).tolist()[0]
        filename = self.save_image_to_disk(img_data)

        data = [
            [id],  # milvus_ids
            [img_name],  # name
            [filename],
            [vectors],  # features
        ]
        collection.insert(data)
        collection.flush()
        print('insert complete')

    def search_data(self, img):
        print('search')
        connections.connect(host=self.host, port=self.port)
        collection = Collection(name=self.field_name)
        collection.load()
        print('连接成功')
        # img = Image.open(image_path)
        vector = img2feature(img).tolist()[0]

        topK = 10
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        res = collection.search(
            [vector],
            "feature",
            search_params,
            topK,
            output_fields=["image_name", "img_data"],
            # expr=exprs
        )
        results_list = []
        for hits in res:
            for hit in hits:
                results_list.append({
                    "image_name": hit.entity.get('image_name'),
                    "filename": hit.entity.get('img_data')  # 获取文件名引用
                })

        return results_list

    def show_nums(self):
        connections.connect(host=self.host, port=self.port)
        collection = Collection(name=self.field_name)
        print('ok')
        print(collection.num_entities)

    def delete_table(self):
        connections.connect(host=self.host, port=self.port)
        default_schema = CollectionSchema(
            fields=self.default_fields, description="test collection")
        collection = Collection(name=self.field_name, schema=default_schema)
        print('>>>', utility.has_collection(self.field_name))
        collection.drop()
        print('>>>', utility.has_collection(self.field_name))
