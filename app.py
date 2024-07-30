import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import io
import requests
import validators
import numpy as np
from numpy.linalg import norm
from PIL import Image
from base64 import b64decode
from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel


class TextEmbeddingRequest(BaseModel):
    texts: List[str] = Field(description="一系列要获取向量表征的文本。")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texts": [
                        "葡萄", "苹果", "栗子"
                    ]
                }
            ]
        }
    }

class ImageEmbeddingRequest(BaseModel):
    images: List[str] = Field(description="一系列要获取向量表征的图片，以Base64或者图片URL链接表示。")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "images": [
                        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAGCAYAAADgzO9IAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAhGVYSWZNTQAqAAAACAAFARIAAwAAAAEAAQAAARoABQAAAAEAAABKARsABQAAAAEAAABSASgAAwAAAAEAAgAAh2kABAAAAAEAAABaAAAAAAAAASAAAAABAAABIAAAAAEAA6ABAAMAAAABAAEAAKACAAQAAAABAAAABqADAAQAAAABAAAABgAAAAA+lA5zAAAACXBIWXMAACxLAAAsSwGlPZapAAACymlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNi4wLjAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyIKICAgICAgICAgICAgeG1sbnM6ZXhpZj0iaHR0cDovL25zLmFkb2JlLmNvbS9leGlmLzEuMC8iPgogICAgICAgICA8dGlmZjpZUmVzb2x1dGlvbj4yODg8L3RpZmY6WVJlc29sdXRpb24+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDx0aWZmOlhSZXNvbHV0aW9uPjI4ODwvdGlmZjpYUmVzb2x1dGlvbj4KICAgICAgICAgPHRpZmY6T3JpZW50YXRpb24+MTwvdGlmZjpPcmllbnRhdGlvbj4KICAgICAgICAgPGV4aWY6UGl4ZWxYRGltZW5zaW9uPjIzPC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj4yMzwvZXhpZjpQaXhlbFlEaW1lbnNpb24+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgpzk9pDAAAAjUlEQVQIHQ3OvQqCYBhA4fOa/WEJBbWIewRCW1vX0SW1diFN7d6EUwkNlRARX4M/mJFv33x44IhmsdKUUBu0uCLBBtw+Lt8K0j3wQ4ZLeJ9gFtmQ38DxaKMdpRfSNWcG5R2HxwH8NTJfoK8nTd6zuLWhM4EiQUyKH4SMpwLZEdF4q2gN7QdGKztxsaLiD0IoMz7JAJrKAAAAAElFTkSuQmCC",
                        "http://images.cocodataset.org/val2017/000000039769.jpg"
                    ]
                },
            ]
        }
    }

class CosineSimilarityRequest(BaseModel):
    vector1: List[float] = Field(description="向量数组")
    vector2: List[float] = Field(description="向量数组")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "vector1": [ 0.15, 0.25, 0.33],
                    "vector2": [ 0.37, 0.21, 0.04]
                },
            ]
        }
    }

class ImageRankingRequest(BaseModel):
    text: str = Field(description="文本")
    images: List[str] = Field(description="要排序的图片列表，Base64形式或者提供网络图片的URL地址")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "水果",
                    "images": [
                        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAGCAYAAADgzO9IAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAhGVYSWZNTQAqAAAACAAFARIAAwAAAAEAAQAAARoABQAAAAEAAABKARsABQAAAAEAAABSASgAAwAAAAEAAgAAh2kABAAAAAEAAABaAAAAAAAAASAAAAABAAABIAAAAAEAA6ABAAMAAAABAAEAAKACAAQAAAABAAAABqADAAQAAAABAAAABgAAAAA+lA5zAAAACXBIWXMAACxLAAAsSwGlPZapAAACymlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNi4wLjAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyIKICAgICAgICAgICAgeG1sbnM6ZXhpZj0iaHR0cDovL25zLmFkb2JlLmNvbS9leGlmLzEuMC8iPgogICAgICAgICA8dGlmZjpZUmVzb2x1dGlvbj4yODg8L3RpZmY6WVJlc29sdXRpb24+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDx0aWZmOlhSZXNvbHV0aW9uPjI4ODwvdGlmZjpYUmVzb2x1dGlvbj4KICAgICAgICAgPHRpZmY6T3JpZW50YXRpb24+MTwvdGlmZjpPcmllbnRhdGlvbj4KICAgICAgICAgPGV4aWY6UGl4ZWxYRGltZW5zaW9uPjIzPC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj4yMzwvZXhpZjpQaXhlbFlEaW1lbnNpb24+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgpzk9pDAAAAjUlEQVQIHQ3OvQqCYBhA4fOa/WEJBbWIewRCW1vX0SW1diFN7d6EUwkNlRARX4M/mJFv33x44IhmsdKUUBu0uCLBBtw+Lt8K0j3wQ4ZLeJ9gFtmQ38DxaKMdpRfSNWcG5R2HxwH8NTJfoK8nTd6zuLWhM4EiQUyKH4SMpwLZEdF4q2gN7QdGKztxsaLiD0IoMz7JAJrKAAAAAElFTkSuQmCC",
                        "http://images.cocodataset.org/val2017/000000039769.jpg"
                    ]
                },
            ]
        }
    }

class TextRankingRequest(BaseModel):
    image: str = Field(description="Base64形式的图片或者网络图片的URL地址")
    texts: List[str] = Field(description="要排序的文本列表")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
                    "texts": [
                        "葡萄", "苹果", "栗子"
                    ]
                },
            ]
        }
    }


# Allowed origins
origins = [
    "*",
    "http://localhost",
    "https://p5js.org",
    "https://editor.p5js.org",
    "https://preview.p5js.org",
]

app = FastAPI(
    title="CLIP API",
    description="使用CLIP模型来将图片、文本转化为向量，并提供度量相似度等功能。"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
os.environ['HF_HOME'] = "/model"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/model")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/model")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/model")

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def process_images(image_strings):
    images = []

    for image_string in image_strings:
        if validators.url(image_string):
            images.append(Image.open(requests.get(image_string, stream=True).raw))
        elif image_string.startswith('data:image'):
            image_type, image_data = image_string.split(';')
            encoding, raw = image_data.split(',')
            images.append(Image.open(io.BytesIO(b64decode(raw))))
        else:
            pass

    return images


def cosine_distance(a, b):
    return np.dot(a, b)/ (norm(a) * norm(b))


@app.get("/", response_class=HTMLResponse, summary="API文档首页")
def read_root():
    return """
    <html>
        <head>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/purecss@3.0.0/build/pure-min.css" integrity="sha384-X38yfunGUhNzHpBaEBsWLO+A0HDYOQi8ufWDkZ0k9e0eXz/tH3II7uKZ9msv++Ls" crossorigin="anonymous">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
            <h1>CLIP API 正在运行</h1>
            <a class="pure-button pure-button-primary" href="/docs">查看文档</a>
        </body>
    </html>
    """

@app.post(
    "/embedding/text", 
    summary="获取文本的向量表征", 
    description="用户可以提供一个包含文本的数组（texts），API将返回这些文本的向量表征。"
)
def embed_text(request: TextEmbeddingRequest):
    inputs = tokenizer(request.texts, return_tensors="pt", padding=True)
    outputs = model.get_text_features(**inputs)
    vectors = outputs.tolist()
    return [
        {'text': request.texts[index], 'vector': vectors[index]}
        for index in range(len(request.texts))
    ]


@app.post(
    "/embedding/image", 
    summary="获取图片的向量表征", 
    description="用户可以提供Base64格式字符串表示的图片（images）或网络图片的URL地址（urls），API将返回这些图片的向量表征。"
)
def embed_image(request: ImageEmbeddingRequest):
    images = process_images(request.images)

    inputs = processor(images=images, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    vectors = outputs.tolist()
    return [
        {'image': request.images[index], 'vector': vectors[index]}
        for index in range(len(request.images))
    ]


@app.post(
    "/similarity/cosine", 
    summary="度量两个向量的Cosine相似度", 
    description="用户可以提供两个向量，API将返回这两个向量的Cosine相似度。"
)
def cosine_similarity(request: CosineSimilarityRequest):
    return cosine_distance(
        np.array(request.vector1),
        np.array(request.vector2)
    )


@app.post(
    "/ranking/images", 
    summary="将图片根据与一段文本的相关性排序", 
    description="用户可以给出一段文本和一系列图片，API将返回图片与文本的相关性排序及具体相似度数值。"
)
def rank_images(request: ImageRankingRequest):
    images = process_images(request.images)

    inputs = processor(text=request.text, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text
    probs = logits_per_text.softmax(dim=1).flatten().tolist()
    
    results = [
        {'image': request.images[index], 'probability': probs[index]}
        for index in range(len(request.images))
    ]
    results.sort(reverse=True, key=lambda x: x['probability'])

    return results


@app.post(
    "/ranking/texts", 
    summary="将文本根据与一张特定图片的相关性排序", 
    description="用户可以给出一张图片和一系列文本，API将返回文本与图片的相关性排序及具体相似度数值。"
)
def rank_texts(request: TextRankingRequest):
    image = process_images([request.image])[0]

    inputs = processor(text=request.texts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).flatten().tolist()
    
    results = [
        {'image': request.texts[index], 'probability': probs[index]}
        for index in range(len(request.texts))
    ]
    results.sort(reverse=True, key=lambda x: x['probability'])

    return results

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

# print(probs)