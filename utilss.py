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

    def insert_data(self, id, img, img_name):
        connections.connect(host=self.host, port=self.port)
        default_schema = CollectionSchema(
            fields=self.default_fields, description="test collection")
        collection = Collection(name=self.field_name, schema=default_schema)

        # Convert img_data (in bytes format) to an image

        vectors = img2feature(img).tolist()[0]

        data = [
            [id],  # milvus_ids
            [img_name],  # name
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
            output_fields=["image_name"],
            # expr=exprs
        )
        results_list = []
        for hits in res:
            for hit in hits:
                results_list.append({
                    "image_name": hit.entity.get('image_name')
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
