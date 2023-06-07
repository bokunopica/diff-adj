### 数据集
疾病统计

### 训练过程
lr=1e-4, ts=12.5k, data=30k
1. 微调Stable Diffusion
2. 换TextEncoder RadBert
3. 换TextEncoder CXR-BERT-specialized
4. 换TextEncoder MedCLIP

5. *增加私有数据 再前三者中最好的模型上喂




### 评估
生成1k图片 根据p19数据集生成图像
1. 给医生看 p19原始图像和生成图像让医生打分（打分规则）
    - 0~1分 最小间隔0.1分 0分完全不真实 1分真实
<!-- 2. densenet121  AUC ACCURACY F1-SCORE -->


### 实验流程
0. 数据预处理
1. 数据集统计
2. 训练模型1.2.3.4
3. 医生简单评估模型1.2.3.4 选取最好的来增加私有数据训练
4. 用方法1、2来评估私有数据的训练前后的模型效果


### 论文数据展示
1. 症状-真实图像和生成图像对
2. macro-averaged CheXpert@10 scores
3. 医生评分表