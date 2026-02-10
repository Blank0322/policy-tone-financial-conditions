
## Authority Media Tone and Financial Condition Forecasting: A Weakly Supervised Learning Measure Based on *People's Daily*
## 权威媒体语调与金融条件预测：基于《人民日报》的弱监督学习测度

**Author/作者:** 0xBlank

---

### 📝 Abstract / 摘要

**English:**  
This research constructs a monthly policy communication index based on the full-text corpus of *People's Daily* and tests its incremental predictive power for changes in financial conditions. To balance economic interpretability with algorithmic generalization, we introduce a **Weakly Supervised Learning** framework. By utilizing dictionary priors to generate "weak labels" for training a classification model, we overcome the coverage limitations of traditional dictionary methods and the semantic ambiguity of unsupervised learning. Using a rigorous **Rolling Out-of-Sample (OOS)** framework and **Clark-West tests**, we find that the weakly supervised index (`tone_logit`) exhibits significant marginal predictability for short-term interest rates (Shibor 3M) in the pre-COVID sample, outperforming both dictionary-based and PCA-based indices.

**中文:**  
本文基于《人民日报》全量文本语料，构建月度政策沟通指数并检验其对金融条件变化的预测增量。为兼顾经济可解释性与算法泛化能力，本文创新性地引入**弱监督学习（Weakly Supervised Learning）**框架，利用词典先验生成“弱标签”以训练分类模型，有效克服了传统词典法的覆盖不足与无监督学习的语义含混。区别于常规样本内回归，本文采用严格的**滚动样本外预测（Rolling OOS）**与 **Clark-West 检验**。实证显示，弱监督指数（`tone_logit`）在疫情前子样本中对短端利率（Shibor 3M）具有显著的边际预测能力，表现优于传统的词典法及 PCA 指数。

---

### 🚀 Key Features / 核心亮点

1.  **Weakly Supervised NLP:** Combines "Expert Rules" (Dictionary) with "Data-Driven" (Logistic Regression) to capture nuanced policy shifts.
    *   **弱监督学习文本处理：** 结合“专家规则”（词典）与“数据驱动”（逻辑回归），捕捉更细致的政策转向。
2.  **Rigorous Econometrics:** Moving beyond in-sample significance to emphasize Out-of-Sample (OOS) predictability with strict information set constraints.
    *   **严格的计量检验：** 不止步于样本内显著性，强调严格信息集约束下的样本外预测（OOS）能力。
3.  **High-Frequency Signal:** Proves that authoritative media tone provides "Alpha" information beyond historical macro variables.
    *   **高频宏观信号：** 证明权威媒体语调蕴含了传统宏观变量历史信息之外的“增量信息”。
4.  **Boundary Conditions:** Discusses why linear models (ARX) outperform complex non-linear models (XGBoost) in small-N, high-noise macro settings.
    *   **适用边界探讨：** 讨论了在“小样本、高噪声”的宏观场景下，为何线性动态模型优于复杂的非线性模型。

---

### 📂 Repository Structure / 仓库结构

```text
├── paper_assets/
│   ├── tables/          # Regression results and summary tables
│   ├── fig/             # Visualization of indices and event studies
│   └── notes/           # Config files (e.g., run_config.json)
├── data/                # (Placeholder) Processed panel data
├── scripts/             # Core processing and analysis scripts
└── README.md
```

---

### 🛠 Reproducibility / 复现指南

#### 1. Environment / 环境配置
- **Python:** 3.x
- **Packages:** `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `pyarrow`, `polars`, `jieba`, `matplotlib`, `seaborn`
- **Random seed:** 42

#### 2. Suggested Run Order / 建议运行顺序
To replicate the results from raw data to final evaluation, please follow this sequence:
请按照以下顺序执行脚本以复现从原始数据到最终评估的全部结果：

1.  **Macro Data:** `python step1_raw_data.py` (Download/prepare macro variables)
2.  **NLP Pipeline:** 
    - `python process_corpus.py` (Text cleaning & temporal aggregation)
    - `python fix_dict.py` (Generate dictionary-based index)
    - `python generate_pca.py` & `python align_pca.py` (Generate unsupervised factors)
    - `python generate_logit.py` (Train Weakly Supervised model & generate `tone_logit`)
3.  **Analysis:**
    - `python step2_panel.py` (Align frequency & build final panel)
    - `python task2_4_oos_final_v2.py` (Rolling OOS & Clark-West evaluation)
4.  **Reporting:** 
    - `python task5_leaderboard.py` (Generate performance rankings)
    - `python step5_finalize_assets.py` (Export tables and figures)

#### 3. Outputs / 输出结果
- **Tables:** `paper_assets/tables`
- **Figures:** `fig/`, `paper_assets/fig`
- **Logs/Notes:** `paper_assets/notes`

---

### 📊 Results Snapshot / 结果速览

| Index Type | OOS Improvement (%) | Clark-West p-value |
| :--- | :--- | :--- |
| **Weakly Supervised (`tone_logit`)** | **1.0205%** | **0.1022** |
| Dictionary (`tone_dict`) | -0.9630% | 0.3444 |
| Unsupervised (`tone_pca`) | -2.5932% | 0.8575 |

![Uploading image.png…]()


*Note: Only the weakly supervised measure provides positive marginal predictive value for short-term interest rates.*
*注：仅弱监督测度为短端利率变化提供了正向的边际预测价值。*

---

### ✉️ Contact / 联系方式
For questions regarding the methodology or code, please open an issue or contact **0xBlank**.
如有关于方法论或代码的疑问，请提交 Issue 或联系作者 **0xBlank**。

---
*Disclaimer: This repository is for academic sharing only. The copyright of the original text corpus belongs to the respective publisher.*
*免责声明：本仓库仅供学术交流使用，原始文本语料版权归相关报社所有。*
