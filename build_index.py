#!/usr/bin/env python3
"""
CLIP索引构建脚本 (修正增强版)
功能：
1. 构建全局图像和文本索引 (保持原有搜索功能)
2. 额外构建分片索引 (用于解决类别不平衡的推荐问题)
"""

import os
import argparse
import pandas as pd
import numpy as np
import faiss
import pickle
from src.clip_matcher import CLIPMatcher


def build_indexes(image_dir: str = "test_Images",
                  captions_file: str = "test_styles.csv",
                  model_path: str = "./clip-vit-base-patch32"):
    print("=" * 60)
    print("🚀 CLIP索引构建工具 (增强版)")
    print("=" * 60)

    # --- 1. 基础检查 ---
    if not os.path.exists(image_dir):
        print(f"❌ 错误: 图像目录 '{image_dir}' 不存在")
        return False
    if not os.path.exists(captions_file):
        print(f"❌ 错误: 描述文件 '{captions_file}' 不存在")
        return False
    if not os.path.exists(model_path):
        print(f"❌ 错误: CLIP模型路径 '{model_path}' 不存在")
        return False

    try:
        # --- 2. 初始化 ---
        print(f"📦 正在初始化CLIP匹配器...")
        matcher = CLIPMatcher(model_path=model_path)

        # --- 3. 构建标准全局索引 (保留原逻辑，确保基础搜索可用) ---
        print(f"\n[1/3] 🖼️  正在构建全局图像索引...")
        matcher.build_image_index(image_dir, "image_index.pkl")

        print(f"\n[2/3] 📝 正在构建全局文本索引...")
        matcher.build_text_index(captions_file, "text_index.pkl")

        # --- 4. 构建分片索引 (新增逻辑：复用已有特征，高效拆分) ---
        print(f"\n[3/3] 🍰 正在构建分片索引 (解决类别不平衡)...")
        matcher.build_partition_index(captions_file)

        print("\n" + "=" * 60)
        print("✅ 所有索引构建完成!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ 索引构建失败: {e}")
        # 打印详细错误栈，方便调试
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数 (修正了索引检查逻辑)"""
    parser = argparse.ArgumentParser(description="构建CLIP索引")
    parser.add_argument("--image_dir", default="test_Images")
    parser.add_argument("--captions_file", default="test_styles.csv")
    parser.add_argument("--model_path", default="./clip-vit-base-patch32")
    parser.add_argument("--force", action="store_true", help="强制重新构建")

    args = parser.parse_args()

    # 定义所有必须存在的索引文件
    required_indexes = [
        "image_index.pkl",
        "text_index.pkl",
        "index_apparel.pkl",
        "index_footwear.pkl",
        "index_others.pkl"
    ]

    # 检查缺失的文件
    missing_files = [f for f in required_indexes if not os.path.exists(f)]

    should_build = True

    if not args.force:
        if not missing_files:
            # 1. 情况A：所有文件都齐全
            print("✅ 检测到所有索引文件均已存在。")
            response = input("是否强制重新构建？(y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("取消构建，直接退出。")
                should_build = False

        elif os.path.exists("image_index.pkl") and missing_files:
            # 2. 情况B：有旧索引，但缺新分片 (典型的升级场景)
            print("⚠️  检测到存在旧版索引，但缺失推荐系统所需的分片索引：")
            print(f"   缺失: {missing_files}")
            response = input("为了启用推荐功能，强烈建议重新构建。是否继续？(Y/n): ")
            # 这里的逻辑是：默认回车(Y)或者是y都继续，只有明确输n才退出
            if response.lower() in ['n', 'no']:
                print("⚠️  警告：您跳过了构建，推荐系统可能无法工作！")
                should_build = False

        # 3. 情况C：什么都没有 -> 直接构建，不询问

    if should_build:
        success = build_indexes(
            image_dir=args.image_dir,
            captions_file=args.captions_file,
            model_path=args.model_path
        )

        if success:
            print("\n🎉 索引就绪! 现在请运行 'python main.py' 启动应用。")


if __name__ == "__main__":
    main()