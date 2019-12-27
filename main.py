import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__':
    root = "E:\\DATA\\tianchi\\chongqing1_round1_train1_20191223"
    image_file_path = os.path.join(root, "images")
    labels = json.load(open(os.path.join(root, "annotations.json"), "rb"))
    images = labels["images"]
    images_id_path_dict = {image["id"]:{"file_name":image["file_name"]} for image in images}
    categories = labels["categories"]
    categories_id_ = {categorie["id"]:categorie["name"] for categorie in categories}
    annotations = labels["annotations"]
    image_ids = []

    for annotation in annotations:
        image_id = annotation["image_id"]
        if "annotations" in images_id_path_dict[image_id].keys():
            images_id_path_dict[image_id]["annotations"].append(annotation)
        else:
            images_id_path_dict[image_id]["annotations"] = [annotation]

    for images_id, info in images_id_path_dict.items():
        annotations = info["annotations"]
        file_name = info["file_name"]
        image_path = os.path.join(image_file_path, file_name)
        image = cv2.imread(image_path)
        print(len(annotations))
        if len(annotations)<5:
            continue
        for annotation in annotations:
            bbox = annotation["bbox"]
            category_id = annotation["category_id"]
            area = annotation["area"]
            iscrowd = annotation["iscrowd"]
            bbox = [int(i+0.5) for i in bbox]
            pilimg = Image.fromarray(image)
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            font = ImageFont.truetype("simhei.ttf", 14, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
            draw.text((bbox[0], bbox[1]-14), str(categories_id_[category_id]), (0, 255, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
            image = np.array(pilimg)
            # cv2.putText(image, str(category_id), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255))
        cv2.imshow("image", image)
        cv2.waitKey(0)