#!/usr/bin/env python3
"""
CLIP索引构建脚本
用于预先构建图像和文本的FAISS索引，提高应用启动速度
"""

import os
import argparse
from src.clip_matcher import CLIPMatcher


def build_indexes(image_dir: str = "test_Images",
                 captions_file: str = "test_styles.csv",
                 model_path: str = "./clip-vit-base-patch32"):
    """
    构建CLIP索引
    
    Args:
        image_dir: 图像目录路径
        captions_file: 图像描述文件路径
        model_path: CLIP模型路径
    """
    print("=" * 60)
    print("🚀 CLIP索引构建工具")
    print("=" * 60)
    
    # 检查输入文件和目录
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
        # 初始化CLIP匹配器
        print(f"📦 正在初始化CLIP匹配器...")
        matcher = CLIPMatcher(model_path=model_path)
        
        # 构建图像索引
        print(f"\n🖼️  正在构建图像索引...")
        print(f"   图像目录: {image_dir}")
        matcher.build_image_index(image_dir, "image_index.pkl")
        
        # 构建文本索引
        print(f"\n📝 正在构建文本索引...")
        print(f"   描述文件: {captions_file}")
        matcher.build_text_index(captions_file, "text_index.pkl")
        
        print("\n" + "=" * 60)
        print("✅ 索引构建完成!")
        print("   - image_index.pkl: 图像索引文件")
        print("   - text_index.pkl: 文本索引文件")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 索引构建失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建CLIP索引")
    parser.add_argument(
        "--image_dir", 
        default="test_Images",
        help="图像目录路径 (默认: test_Images)"
    )
    parser.add_argument(
        "--captions_file",
        default="test_styles.csv",
        help="图像描述文件路径 (默认: test_styles.csv)"
    )
    parser.add_argument(
        "--model_path",
        default="./clip-vit-base-patch32",
        help="CLIP模型路径 (默认: ./clip-vit-base-patch32)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新构建索引（即使已存在）"
    )
    
    args = parser.parse_args()
    
    # 检查是否已存在索引文件
    if not args.force:
        if os.path.exists("image_index.pkl") and os.path.exists("text_index.pkl"):
            response = input("索引文件已存在，是否重新构建？(y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("取消构建")
                return
    
    # 构建索引
    success = build_indexes(
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        model_path=args.model_path
    )
    
    if success:
        print("\n🎉 现在可以运行 'python gradio_app.py' 启动应用!")
    else:
        print("\n💥 索引构建失败，请检查错误信息")


if __name__ == "__main__":
    main() 