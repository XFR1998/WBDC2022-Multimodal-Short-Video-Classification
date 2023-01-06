# 项目介绍
- 1.此仓库代码为2022-微信大数据挑战赛-初赛提交代码  
- 2.复赛方案有关介绍可参考链接：https://blog.csdn.net/weixin_43646592/article/details/126904740
- 3.😄参赛队员：Furen Xu (代号：苦行僧、尴尬、鸡哥), Jinbo Huang (代号：波神、波波、波的哥)


# 依赖环境
- python：3.7.0
- CUDA：11.1.1
- 操作系统：20.04.1-Ubuntu  x86_64 x86_64 x86_64 GNU/Linux


# 代码结构
```
./
├── README.md
├── requirements.txt      # Python包依赖文件 
├── init.sh               # 初始化脚本，用于准备环境
├── train.sh              # 模型训练脚本
├── inference.sh          # 模型测试脚本 
├── src/                  # 核心代码
│   ├── ALBEF/      # 模型A
│   ├── MLM_MFM/        # 模型B
│   ├── MLM_MFM_ITM/  # 模型C
│   ├── kfold_logits/      
│   ├── inference_many_kfold_models.py
│   ├── 切分K折数据集.py
├── data/
│   ├── annotations/      # 数据集标注（仅作示意，无需上传）
│   ├── zip_feats/        # 数据集视觉特征（仅作示意，无需上传）
│   ├── kfold_data/
```

# 算法模型介绍
### 模型A
- 模型源自：https://github.com/salesforce/albef
- 采用类似ALBEF的结构（双流），bert前6层进行文本学习，后6层和视频特征做cross-attention融合。  
- 预训练任务（10 epoch）：mlm, mfm, itm  
- 采用五折微调训练  

### 模型B
- 模型源自：https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st
- 采用类似上述链接的模型结构（单流），在embedding层：其输入以这种形式拼接: [CLS] Video_frame [SEP] Video_title [SEP]。    
- 预训练任务（20 epoch）：mlm, mfm 
- 采用五折微调训练 

### 模型C
- 模型源自：https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st
- 采用类似上述链接的模型结构（单流），在embedding层：其输入以这种形式拼接: [CLS] Video_frame [SEP] Video_title [SEP]。    
- 预训练任务（20 epoch）：mlm, mfm, itm  
- 采用五折微调训练 



# 运行流程：
运行以下指令，安装依赖包，对训练集切分成五折数据集放在data目录下：  
`init.sh`

运行以下指令，依次预训练A，B，C个模型，然后分别加载预训练最优模型进行5折微调训练：  
`train.sh`


运行以下指令，依次使用A，B，C个模型在b榜测试的结果（并保存各自的logits）, 最后A，B，C这3个模型的logits相加取平均再argmax预测b榜测试的结果，保存在data/result.csv：  
`inference.sh`  



# 初赛B榜测试结果：
线上最高分：0.700648  
（category1_f1_macro：0.772157，category1_f1_micro：0.78468，category2_f1_macro：0.570596，category2_f1_micro：0.67516）

# 开源预训练模型：
采用huggingface上的：hfl/chinese-roberta-wwm-ext  
链接：https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main