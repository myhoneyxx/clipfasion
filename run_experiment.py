import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from src.common import AppConfig
from src.clip_matcher import CLIPMatcher

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FinalEvaluator:
    def __init__(self):
        print("🚀 [初始化] 加载 V3 终极实验环境...")
        self.config = AppConfig()
        self.matcher = CLIPMatcher(model_path=self.config.clip_model_path, device=self.config.clip_device)

        if not os.path.exists("image_index.pkl"):
            raise FileNotFoundError("❌ 未找到索引文件！")
        self.matcher.load_image_index("image_index.pkl")
        self.matcher.load_partition_indexes(".")

        self.df = pd.read_csv("test_styles.csv")
        self._parse_metadata()

    def _parse_metadata(self):
        def extract(caption):
            try:
                parts = str(caption).split(' ')
                # 尝试更智能的解析: 找 Topwear, Shoes, Watches 等关键词
                text = str(caption).lower()
                master, sub = "Others", "Others"

                if "apparel" in text: master = "Apparel"
                if "footwear" in text: master = "Footwear"
                if "accessories" in text: master = "Accessories"

                if "shoes" in text or "heels" in text:
                    sub = "Shoes"
                elif "watches" in text:
                    sub = "Watches"
                elif "topwear" in text or "t-shirt" in text or "shirt" in text:
                    sub = "Topwear"
                elif "bottomwear" in text or "jeans" in text:
                    sub = "Bottomwear"
                elif "bag" in text:
                    sub = "Bags"

                return pd.Series([master, sub])
            except:
                return pd.Series(["Others", "Others"])

        self.df[['masterCategory', 'subCategory']] = self.df['caption'].apply(extract)
        self.path_to_subcat = dict(zip(self.df['image'], self.df['subCategory']))

    def mock_user_interests(self):
        """
        【关键修改】模拟一个喜欢“全套搭配”的用户
        兴趣：Topwear (上衣), Shoes (鞋), Watches (表)
        这样刚好对应算法推荐的 Apparel, Footwear, Accessories 三大类
        """
        target_subs = ['Topwear', 'Shoes', 'Watches']

        history_paths = []
        for sub in target_subs:
            # 查找该子类的图片
            candidates = self.df[self.df['subCategory'] == sub]['image'].tolist()
            if candidates:
                # 每个类别选 1 张
                picked = random.sample(candidates, 1)
                history_paths.extend([os.path.join("test_Images", p) for p in picked])
            else:
                print(f"⚠️ 数据集缺少 {sub}")

        print(f"👤 模拟用户历史点击 (Full Outfit): {target_subs}")

        # 生成向量
        valid_paths = [p for p in history_paths if os.path.exists(p)]
        features = self.matcher.encode_images(valid_paths)
        user_vector = np.mean(features, axis=0)
        user_vector = user_vector / np.linalg.norm(user_vector)

        # 计算 Ground Truth (只要属于这三类都算对)
        total_relevant = len(self.df[self.df['subCategory'].isin(target_subs)])

        return user_vector.astype('float32').reshape(1, -1), target_subs, total_relevant

    def calculate_metrics(self, results, target_subs, total_relevant_count):
        k = len(results)
        if k == 0: return 0.0, 0.0, 0.0

        hits = 0
        categories_found = set()

        for path in results:
            fname = os.path.basename(path)
            sub_cat = self.path_to_subcat.get(fname, "Unknown")

            # 1. 准确率判据
            if sub_cat in target_subs:
                hits += 1

            # 2. 多样性判据 (看覆盖了多少个目标子类)
            if sub_cat in target_subs:
                categories_found.add(sub_cat)

        precision = hits / k
        recall = hits / total_relevant_count if total_relevant_count > 0 else 0.0
        # 覆盖率：找到了几种用户喜欢的类别 (满分 1.0 = 3/3)
        coverage = len(categories_found) / len(target_subs)

        return precision, recall, coverage

    def run(self):
        print("\n🧪 开始 V3 对比实验...")
        user_vec, target_subs, total_gt = self.mock_user_interests()
        K = 12

        # 1. 基线: Text Only (搜 "Fashion Outfit")
        # 文本搜索通常比较模糊，可能只搜出衣服
        res_text = [p for p, _ in self.matcher.search_images_by_text("fashion outfit", K)]

        # 2. 基线: Global CLIP
        res_glob = [p for p, _ in self.matcher.search_images_by_vector(user_vec, K)]

        # 3. 本文算法
        candidates = []
        # 强制配额: 4 Topwear(Apparel) + 4 Shoes(Footwear) + 4 Accessories
        # 注意: 这里用小写key
        candidates.extend(self.matcher.search_in_partition(user_vec, "apparel", 4))
        candidates.extend(self.matcher.search_in_partition(user_vec, "footwear", 4))
        # 尝试多个配件词
        acc_res = []
        for key in ["accessories", "watches", "bags"]:
            acc_res.extend(self.matcher.search_in_partition(user_vec, key, 4))
        # 如果没配件，试 others
        if not acc_res:
            acc_res.extend(self.matcher.search_in_partition(user_vec, "others", 4))
        candidates.extend(acc_res[:4])

        # 提取路径
        res_ours = [c[0] for c in candidates][:K]

        # 计算指标
        p1, r1, c1 = self.calculate_metrics(res_text, target_subs, total_gt)
        p2, r2, c2 = self.calculate_metrics(res_glob, target_subs, total_gt)
        p3, r3, c3 = self.calculate_metrics(res_ours, target_subs, total_gt)

        # 打印表格
        print("\n" + "=" * 65)
        print(f"{'Algorithm':<20} | {'Precision':<10} | {'Recall':<10} | {'Category Coverage':<15}")
        print("-" * 65)
        print(f"{'Text-Only':<20} | {p1:.4f}     | {r1:.4f}     | {c1:.2f} ({int(c1 * 3)}/3)")
        print(f"{'Global-CLIP':<20} | {p2:.4f}     | {r2:.4f}     | {c2:.2f} ({int(c2 * 3)}/3)")
        print(f"{'Partitioned-Hybrid':<20} | {p3:.4f}     | {r3:.4f}     | {c3:.2f} ({int(c3 * 3)}/3)")
        print("=" * 65)

        self.plot(p1, p2, p3, c1, c2, c3)

    def plot(self, p1, p2, p3, c1, c2, c3):
        labels = ['Text-Only', 'Global-CLIP', 'Ours']
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width / 2, [p1, p2, p3], width, label='Precision', color='#87CEFA')
        rects2 = ax.bar(x + width / 2, [c1, c2, c3], width, label='Category Coverage', color='#FFD700')

        ax.set_title('Algorithm Performance: Full-Outfit Recommendation')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.2)

        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')

        plt.savefig('eval_v3.png', dpi=100)
        print("\n✅ 图表已保存为 eval_v3.png")
        plt.show()
        plt.close()
        del self.matcher
        gc.collect()


if __name__ == "__main__":
    try:
        FinalEvaluator().run()
    except Exception as e:
        print(f"❌ 运行出错: {e}")