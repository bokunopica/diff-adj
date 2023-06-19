### 训练过程
lr=1e-4, ts=12.5k, data=30k, batch_size = 1
1. 微调Stable Diffusion [已调通]
2. 换TextEncoder RadBert [已调通]
3. 换TextEncoder CXR-BERT-specialized [已调通]
4. 换TextEncoder MedCLIP [已调通]
    - https://huggingface.co/docs/transformers/model_doc/clip#clip
    - https://zhuanlan.zhihu.com/p/603168346

5. *增加私有数据 再前三者中最好的模型上喂

### 评估
生成1k图片 根据p19数据集生成图像
1. 给医生看 p19原始图像和生成图像让医生打分（打分规则）
    - 0~1分 最小间隔0.1分 0分完全不真实 1分真实
<!-- 2. densenet121  AUC ACCURACY F1-SCORE -->

### 实验流程
0. 数据预处理 [ok]
    - 训练集 p10~p18 所有PA图像
    - 测试集 p19 所有PA图像
1. 数据集统计 [ok]
    - 疾病数量统计 [ok]
        - 训练集
        - 测试集
2. 训练模型1.2.3.4 [ok]
3. 私有数据验证
    - 数据量：1000pairs
    - 数据处理: 
        - impression翻译
        - label提取
        - 分训练验证集6：4
    - 训练
    - 4个模型的验证
        - 生成挑选实验
            - 人工
            - MedCLIP - impression
            - MedCLIP - "A photo of a chest xray"
        - 模型验证 - p19 - 7000+
            - 分类器验证 1000
                - 单疾病分类
                    - 按疾病比例
                    - 每张图生成8张-过MedCLIP选概率最大的-人工过一遍删除离谱的图像
                    - 扔到densenet分类器中得出AUC指标
                - 双疾病分类
                    - 最多的两种复合疾病
                    - 第二多的两种复合疾病
            - AUC指标
    - 最好的模型验证
        - 人类验证 - 私有数据集
            - 微调 6：4
            - 每张图生成8张-过MedCLIP选概率最大的-人工过一遍删除离谱的图像
            - 生成后的数据每个疾病选取3张
                - 没有就pass
            - 每个模型分开打分 评判最优模型

### 论文数据展示
1. 症状-真实图像和生成图像对
2. macro-averaged CheXpert@10 scores
3. 医生评分表

### 后续工作
1. latent channels 4 -> 8
2. rgb channels  3 -> 1
3. 多疾病问题