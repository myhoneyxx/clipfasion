import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


class CLIPMatcher:
    """基于CLIP的文本和图像匹配系统"""

    def __init__(self, model_path: str = "../clip-vit-base-patch32", device: str = None):
        """
        初始化CLIP匹配器

        Args:
            model_path: CLIP模型路径
            device: 计算设备 ('cuda', 'cpu' 或 None 自动选择)
        """
        self.partition_indexes = {}
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

    def build_partition_index(self, captions_file: str):
        """
        辅助函数：基于 CSV 类别信息构建分片索引 (作为 CLIPMatcher 类的方法)

        Args:
            captions_file: 包含 image 和 caption 列的 CSV 文件路径
        """
        print(f"\n[3/3] 🍰 正在构建分片索引 (解决类别不平衡)...")

        # 1. 读取CSV获取类别信息
        try:
            # 确保 pandas 已在文件顶部导入: import pandas as pd
            df = pd.read_csv(captions_file)
        except Exception as e:
            print(f"      ❌ 读取描述文件失败: {e}")
            return

        # 2. 创建文件名到类别的映射字典
        print("      正在解析类别映射...")
        img_category_map = {}
        for _, row in df.iterrows():
            # 确保转为字符串并小写，防止 AttributeError
            fname = str(row['image'])
            caption = str(row['caption']).lower()

            # 简单分类规则
            if "footwear" in caption or "shoes" in caption:
                cat = "footwear"
            elif "apparel" in caption:
                cat = "apparel"
            else:
                cat = "others"
            img_category_map[fname] = cat

        # 3. 准备分桶容器
        partitions = {
            "apparel": {'paths': [], 'features': []},
            "footwear": {'paths': [], 'features': []},
            "others": {'paths': [], 'features': []}
        }

        # 4. 遍历 self 中已经算好的所有图片和特征
        # 修正点：使用 self.image_paths 代替 matcher.image_paths
        if not hasattr(self, 'image_paths') or not self.image_paths:
            print("      ⚠️ 警告: 内存中没有图像特征，请先调用 build_image_index。")
            return

        total_images = len(self.image_paths)  # 👈 已修正为 self

        print(f"      正在对 {total_images} 张图像进行分类拆分...")

        count_hit = 0
        for idx, path in enumerate(self.image_paths):  # 👈 已修正为 self
            filename = os.path.basename(path)
            # 查找该图片的类别，找不到默认为 others
            category = img_category_map.get(filename, "others")

            if category in partitions:
                partitions[category]['paths'].append(path)
                # 修正点：使用 self.image_features 代替 matcher.image_features
                partitions[category]['features'].append(self.image_features[idx])  # 👈 已修正为 self
                count_hit += 1

        # 5. 保存分片索引
        for cat_name, data in partitions.items():
            paths = data['paths']
            feats = data['features']

            if len(paths) > 0:
                # 转换为 FAISS 需要的 float32 numpy 数组
                # 确保 numpy 已导入: import numpy as np
                feats_np = np.array(feats).astype('float32')

                # 构建 FAISS 索引
                # 确保 faiss 已导入
                dimension = feats_np.shape[1]
                sub_index = faiss.IndexFlatIP(dimension)
                sub_index.add(feats_np)

                # 保存为 pkl 文件
                save_path = f"index_{cat_name}.pkl"
                index_data = {
                    'image_paths': paths,
                    'image_features': feats_np,
                    'image_index': faiss.serialize_index(sub_index)
                }

                try:
                    with open(save_path, 'wb') as f:
                        # 确保 pickle 已导入
                        pickle.dump(index_data, f)
                    print(f"      ✅ 已保存分片: {save_path} (包含 {len(paths)} 条)")
                except Exception as e:
                    print(f"      ❌ 保存分片 {save_path} 失败: {e}")

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

    def load_partition_indexes(self, index_dir="."):
        """
        加载所有 index_xxx.pkl 分片文件到内存

        Args:
            index_dir: 索引文件所在的目录

        Returns:
            bool: 是否成功加载了至少一个分片
        """
        if not os.path.exists(index_dir):
            print(f"❌ 索引目录不存在: {index_dir}")
            return False

        count = 0
        # 遍历目录寻找 index_*.pkl
        for filename in os.listdir(index_dir):
            # 严格匹配文件名模式，排除 image_index.pkl (全局索引) 和 text_index.pkl
            if filename.startswith("index_") and filename.endswith(".pkl"):
                # 提取类别名: index_apparel.pkl -> apparel
                cat = filename.replace("index_", "").replace(".pkl", "")
                file_path = os.path.join(index_dir, filename)

                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

                    # 简单校验数据结构，防止加载损坏文件
                    if 'image_paths' not in data or 'image_index' not in data:
                        print(f"⚠️ 跳过无效索引文件: {filename}")
                        continue

                    # 反序列化并存储
                    # 注意：确保 __init__ 中已经初始化了 self.partition_indexes = {}
                    self.partition_indexes[cat] = {
                        'paths': data['image_paths'],
                        'index': faiss.deserialize_index(data['image_index'])
                    }
                    print(f"✅ 已加载分片索引: {cat} (包含 {len(data['image_paths'])} 条数据)")
                    count += 1
                except Exception as e:
                    print(f"❌ 加载分片 {filename} 失败: {e}")

        return count > 0
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

    def search_in_partition(self, query_vector: np.ndarray, category: str, top_k: int = 5):
        """
        在指定的分片索引中搜索相似图像

        Args:
            query_vector: 查询向量 (numpy array)
            category: 分片类别名称 (如 'apparel', 'footwear')
            top_k: 期望返回的结果数量

        Returns:
            List[Tuple[str, float]]: [(图片路径, 相似度分数), ...]
        """
        # 1. 检查该分片是否存在
        if category not in self.partition_indexes:
            # 如果没有这个类别的索引（比如没有美妆数据），直接返回空，不报错
            return []

        target = self.partition_indexes[category]
        index = target['index']
        paths = target['paths']

        # 2. 预处理向量 (确保是 2D float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype('float32')

        # 3. 智能调整 Top-K
        # 如果请求 5 个结果，但该类别只有 2 张图，则只搜 2 张，防止 FAISS 报错或返回填充值
        real_k = min(top_k, len(paths))
        if real_k == 0:
            return []

        # 4. 执行搜索
        scores, indices = index.search(query_vector, real_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS 可能会在找不到足够结果时返回 -1，必须过滤
            if idx != -1 and idx < len(paths):
                results.append((paths[idx], float(score)))

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