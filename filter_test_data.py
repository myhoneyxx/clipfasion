import pandas as pd

# 1. 配置文件路径（确保正确！）
file_path = "styles.csv"  # 相对路径（代码和文件同目录）
# 绝对路径备用：file_path = "D:\\graduation\\Clip\\Clip\\styles.csv"

# 2. 强制读取为1列（无视任何分隔符）
df_raw = pd.read_csv(
    file_path,
    encoding="utf-8",
    sep="\t\t\t",  # 用3个制表符分隔（文件中没有，所有数据合并为1列）
    header=0,  # 第1行作为表头（后续丢弃）
    on_bad_lines="skip",
    engine="python",
    names=["all_data"]  # 合并后的列名
)

# 3. 丢弃表头行，只保留数据行
df_data = df_raw.iloc[1:].copy()  # 跳过第1行（表头）
df_data = df_data[df_data["all_data"].notna()]  # 过滤空行
df_data = df_data[df_data["all_data"].str.strip() != ""]  # 过滤纯空字符串行

print("=" * 60)
print(f"✅ 读取数据：共 {len(df_data)} 条有效数据（已过滤表头/空行）")
print("前3条原始数据（合并为1列）：")
for i, (_, row) in enumerate(df_data.head(3).iterrows()):
    print(f"  数据{i + 1}：{row['all_data']}")


# 4. 拆分函数（按逗号拆分，提取10列数据）
def split_data(data_str):
    data_str = str(data_str).strip()
    if not data_str:
        return [""] * 10  # 空数据返回10个空字符串

    # 按逗号拆分，去空格并过滤空值
    parts = [p.strip() for p in data_str.split(",") if p.strip()]
    # 确保拆分后至少有10列（不足补空）
    parts += [""] * (10 - len(parts))
    return parts[:10]  # 只取前10列（对应：id, gender, masterCategory, ..., productDisplayName）


# 5. 应用拆分函数，得到10列数据
df_split = df_data["all_data"].apply(split_data).apply(pd.Series)
# 给拆分后的列命名
df_split.columns = [
    "id", "gender", "masterCategory", "subCategory",
    "articleType", "baseColour", "season", "year",
    "usage", "productDisplayName"
]

# 6. 合并多列作为完整描述（gender 到 productDisplayName）
# 定义要合并的列（按顺序）
description_cols = [
    "gender", "masterCategory", "subCategory",
    "articleType", "baseColour", "season", "year",
    "usage", "productDisplayName"
]

# 合并列：用空格连接非空值，避免多余分隔符
df_split["full_caption"] = df_split[description_cols].apply(
    lambda row: " ".join([str(val) for val in row if val.strip() != ""]),
    axis=1
)

# 7. 清理核心数据（id + 完整描述）
df_core = df_split[["id", "full_caption"]].copy()
# 清理 id 列（纯数字字符串，匹配图像文件名）
df_core["id"] = df_core["id"].astype(str).str.strip()
df_core = df_core[df_core["id"].str.isdigit()]  # 只保留纯数字 id
# 清理描述列（过滤空描述）
df_core = df_core[df_core["full_caption"].str.strip() != ""]
# 去重（避免重复 id）
df_core = df_core.drop_duplicates(subset=["id"], keep="first")

# 8. 最终结果展示
print("\n" + "=" * 60)
print("✅ 成功提取核心数据（id + 多列合并描述）！")
print(f"有效数据条数：{len(df_core)} 条")
print("\n前5条核心数据（id + 完整描述）：")
for idx, (_, row) in enumerate(df_core.head(5).iterrows()):
    caption = row["full_caption"]
    # 描述过长时截断，方便查看
    display_caption = caption[:80] + "..." if len(caption) > 80 else caption
    print(f"  样本{idx + 1}：id='{row['id']}' → 描述='{display_caption}'")

print("\n📌 关键统计：")
print(f"  - 唯一 id 数量：{df_core['id'].nunique()} 个")
print(f"  - 最长描述长度：{df_core['full_caption'].str.len().max()} 字符")
print(f"  - 平均描述长度：{round(df_core['full_caption'].str.len().mean(), 2)} 字符")
print("=" * 60)

# 9. 后续：筛选1万条测试数据 + 复制图像（可选，保留原逻辑）
TEST_SIZE = 10000
TEST_IMAGE_DIR = "test_Images"
TEST_CAPTIONS = "test_styles.csv"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

print(f"\n筛选 {TEST_SIZE} 条测试数据...")
sample_size = min(TEST_SIZE, len(df_core))
df_test = df_core.sample(n=sample_size, random_state=42)  # 随机筛选（可复现）
print(f"筛选出 {len(df_test)} 条测试数据")

# 复制测试图像
import os
import shutil
from tqdm import tqdm

print(f"\n正在复制 {len(df_test)} 张测试图像...")
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
missing_images = 0
valid_rows = []

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    image_id = row["id"]
    caption = row["full_caption"]

    # 查找实际存在的图像文件
    found_image_path = None
    image_name_with_ext = None
    for ext in IMAGE_EXTENSIONS:
        raw_image_path = os.path.join("Images", f"{image_id}{ext}")  # 原始图像目录
        if os.path.exists(raw_image_path):
            found_image_path = raw_image_path
            image_name_with_ext = f"{image_id}{ext}"
            break

    if found_image_path:
        # 复制图像到测试目录
        test_image_path = os.path.join(TEST_IMAGE_DIR, image_name_with_ext)
        shutil.copy2(found_image_path, test_image_path)
        valid_rows.append({
            "image": image_name_with_ext,
            "caption": caption
        })
    else:
        missing_images += 1

# 保存测试描述文件
df_valid = pd.DataFrame(valid_rows)
df_valid.to_csv(TEST_CAPTIONS, index=False, encoding='utf-8', quoting=1)

# 最终统计
test_image_count = len(os.listdir(TEST_IMAGE_DIR))
print(f"\n" + "=" * 50)
print("✅ 测试数据集准备完成！")
print(f"📁 测试图像目录：{TEST_IMAGE_DIR}（共 {test_image_count} 张）")
print(f"📄 测试描述文件：{TEST_CAPTIONS}（共 {len(df_valid)} 条）")
print(f"❌ 未找到的图像：{missing_images} 张")
print("=" * 50)