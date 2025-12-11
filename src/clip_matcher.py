import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import faiss
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Union
import pickle
from tqdm import tqdm
import cv2


class CLIPMatcher:
    """基于CLIP的文本和图像匹配系统"""

    def __init__(self, model_path: str = "../clip-vit-base-patch32", device: str = None):
        """
        初始化CLIP匹配器

        Args:
            model_path: CLIP模型路径
            device: 计算设备 ('cuda', 'cpu' 或 None 自动选择)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载CLIP模型和处理器
        print("正在加载CLIP模型...")
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 初始化存储
        self.image_paths = []
        self.image_features = None
        self.text_features = None
        self.captions = []

        # FAISS索引
        self.image_index = None
        self.text_index = None

        print("CLIP匹配器初始化完成!")

    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码图像

        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小

        Returns:
            图像特征向量数组
        """
        features = []

        print(f"正在编码 {len(image_paths)} 张图像...")
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(image)
                except Exception as e:
                    print(f"无法加载图像 {path}: {e}")
                    # 创建一个空白图像作为占位符
                    batch_images.append(Image.new('RGB', (224, 224), color='white'))

            if batch_images:
                with torch.no_grad():
                    inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu().numpy())

        return np.vstack(features) if features else np.array([])

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码文本

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            文本特征向量数组
        """
        features = []

        print(f"正在编码 {len(texts)} 条文本...")
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad():
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                features.append(text_features.cpu().numpy())

        return np.vstack(features) if features else np.array([])

    def build_image_index(self, image_dir: str, save_path: str = "image_index.pkl"):
        """
        构建图像索引

        Args:
            image_dir: 图像目录路径
            save_path: 索引保存路径
        """
        # 获取所有图像路径
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.image_paths = []

        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                self.image_paths.append(os.path.join(image_dir, filename))

        print(f"找到 {len(self.image_paths)} 张图像")

        # 编码图像
        self.image_features = self.encode_images(self.image_paths)

        # 构建FAISS索引
        if len(self.image_features) > 0:
            dimension = self.image_features.shape[1]
            self.image_index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            self.image_index.add(self.image_features.astype('float32'))

            # 保存索引
            index_data = {
                'image_paths': self.image_paths,
                'image_features': self.image_features,
                'image_index': faiss.serialize_index(self.image_index)
            }

            with open(save_path, 'wb') as f:
                pickle.dump(index_data, f)

            print(f"图像索引已保存到 {save_path}")

    def build_text_index(self, captions_file: str, save_path: str = "text_index.pkl"):
        """
        构建文本索引

        Args:
            captions_file: 图像描述文件路径
            save_path: 索引保存路径
        """
        # 读取图像描述
        try:
            df = pd.read_csv(captions_file)
            self.captions = df['caption'].tolist()
            caption_images = df['image'].tolist()
        except Exception as e:
            print(f"读取描述文件失败: {e}")
            return

        print(f"找到 {len(self.captions)} 条图像描述")

        # 编码文本
        self.text_features = self.encode_texts(self.captions)

        # 构建FAISS索引
        if len(self.text_features) > 0:
            dimension = self.text_features.shape[1]
            self.text_index = faiss.IndexFlatIP(dimension)
            self.text_index.add(self.text_features.astype('float32'))

            # 保存索引
            index_data = {
                'captions': self.captions,
                'caption_images': caption_images,
                'text_features': self.text_features,
                'text_index': faiss.serialize_index(self.text_index)
            }

            with open(save_path, 'wb') as f:
                pickle.dump(index_data, f)

            print(f"文本索引已保存到 {save_path}")

    def load_image_index(self, index_path: str = "image_index.pkl"):
        """加载图像索引"""
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)

            self.image_paths = index_data['image_paths']
            self.image_features = index_data['image_features']
            self.image_index = faiss.deserialize_index(index_data['image_index'])

            print(f"已加载图像索引，包含 {len(self.image_paths)} 张图像")
            return True
        except Exception as e:
            print(f"加载图像索引失败: {e}")
            return False

    def load_text_index(self, index_path: str = "text_index.pkl"):
        """加载文本索引"""
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)

            self.captions = index_data['captions']
            self.caption_images = index_data['caption_images']
            self.text_features = index_data['text_features']
            self.text_index = faiss.deserialize_index(index_data['text_index'])

            print(f"已加载文本索引，包含 {len(self.captions)} 条描述")
            return True
        except Exception as e:
            print(f"加载文本索引失败: {e}")
            return False

    # 🚨 NEW FUNCTION: 基于向量的直接搜索接口 (支持用户兴趣向量)
    def search_images_by_vector(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        根据 CLIP 特征向量搜索相似图像 (支持用户兴趣向量)

        Args:
            query_vector: 归一化后的 CLIP 特征向量 (NumPy 数组)
            top_k: 返回前k个结果

        Returns:
            (图像路径, 相似度分数) 的列表
        """
        if self.image_index is None:
            return []

        # 确保向量是 float32 类型，并确保其形状为 (1, dimension)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype('float32')

        # 搜索
        scores, indices = self.image_index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(score)))

        return results

    def search_images_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        根据文本搜索相似图像
        """
        if self.image_index is None:
            return []

        # 编码查询文本
        query_features = self.encode_texts([query_text])

        # 搜索
        scores, indices = self.image_index.search(query_features.astype('float32'), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(score)))

        return results

    def search_images_by_image(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        根据图像搜索相似图像
        """
        if self.image_index is None:
            return []

        # 编码查询图像
        query_features = self.encode_images([query_image_path])

        # 搜索
        scores, indices = self.image_index.search(query_features.astype('float32'), top_k + 1)  # +1 因为可能包含自己

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.image_paths):
                image_path = self.image_paths[idx]
                # 跳过查询图像本身
                if os.path.abspath(image_path) != os.path.abspath(query_image_path):
                    results.append((image_path, float(score)))
                    if len(results) >= top_k:
                        break

        return results

    def describe_image(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        描述图像（找到最相似的文本描述）
        """
        if self.text_index is None:
            return []

        # 编码查询图像
        query_features = self.encode_images([image_path])

        # 搜索最相似的文本
        scores, indices = self.text_index.search(query_features.astype('float32'), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.captions):
                results.append((self.captions[idx], float(score)))

        return results