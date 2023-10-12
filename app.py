from flask import Flask, request, jsonify, send_from_directory
from utilss import MilvusManager
import os
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)
milvus = MilvusManager(field_name="image_search",
                       host="124.222.80.128", port="31367", dim=1000)


def decode_image(img_data):
    base64_data = img_data.split(",")[1]
    img_bytes = base64.b64decode(base64_data)
    return Image.open(BytesIO(img_bytes))


@app.route('/', methods=['GET'])
def index():
    return open('index.html').read()


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


@app.route('/create', methods=['POST'])
def create_table():
    milvus.create_table()
    return jsonify({"message": "表格已创建"}), 200


@app.route('/insert', methods=['POST'])
def insert_data():
    data = request.json
    img_path = data["img_path"]
    img_name = data["img_name"]
    milvus.insert_data(img_path, img_name)
    return jsonify({"message": "数据已插入"}), 200


@app.route('/insert_from_folder', methods=['POST'])
def insert_data_from_folder():
    data = request.json
    imgDataList = data["imgDataList"]
    id = 0
    for item in imgDataList:
        id += 1
        img_name = item["name"]
        img_data = item["data"]
        # img = decode_image(img_data)
        milvus.insert_data(id, img_data, img_name)

    return jsonify({"message": f"从文件夹插入了{len(imgDataList)}张图片"}), 200


@app.route('/search', methods=['POST'])
def search_data():
    data = request.get_json()
    img_data = data["img_data"]
    img = decode_image(img_data)
    results = milvus.search_data(img)
    return jsonify(results), 200


@app.route('/show_nums', methods=['GET'])
def show_nums():
    nums = milvus.show_nums()
    return jsonify({"num_entities": nums}), 200


@app.route('/delete', methods=['DELETE'])
def delete_table():
    milvus.delete_table()
    return jsonify({"message": "表格已删除"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=50033)
