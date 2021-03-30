# NLP-Series-text-cls
本项目在基于RoBERTa进行文本分类的基础上，探讨**数据增强、对抗学习、无监督学习UDA**等的影响。  
详细介绍在知乎博客：https://zhuanlan.zhihu.com/p/339894411
# 运行环境
python：3.6  
pytorch：1.7.0  
transformers：4.4.2  
# 数据
- 今日头条短文本分类，根据新闻标题，分类新闻的类别
- 科大讯飞APP分类，根据APP应用的简介，分类APP的类别
# 运行脚本
有监督（以今日头条短文本分类为例）：
```
python run
--task=yq-sentiment
--train_max_seq_length=512
--model_name_or_path=预训练模型的路径
--per_gpu_train_batch_size=100
--per_gpu_eval_batch_size=100
--learning_rate=1e-5
--linear_learning_rate=1e-3
--num_train_epochs=50
--output_dir="./output"
--weight_decay=0.01
--early_stop=2
```

无监督UDA（以今日头条短文本分类为例）：
```
python run_uda.py
--task=shorttext_zj
--train_max_seq_length=40
--model_name_or_path=预训练模型的路径
--per_gpu_train_batch_size=40
--per_gpu_eval_batch_size=600
--learning_rate=1e-5
--linear_learning_rate=1e-3
--num_train_epochs=20
--output_dir="./output"
--weight_decay=0.01
--tsa=log_schedule
--early_stop=2
--tsa=exp_schedule
```
# 实验结果
详细的实验结果请看上面知乎链接

