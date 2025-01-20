# chrome dino auto player
powered by neural network

#### ⚠️所有python脚本务必运行在自身目录下


### 如何运行

本地测试环境: macos15.2; 外观:浅色; 需要允许程序对键盘和屏幕录制的权限

玩小恐龙请用 <a href='https://chrome-dino-game.github.io'>chrome dino</a>
注意:测试发现不同电脑和浏览器的游戏界面都有所不同.其中在`MacBook safair`和`huawei matebook chrome`的上界面一致,运行结果正常.其余可能会因为界面不同和模型泛化性较差原因,无法正常跳跃

打开`auto_dino`,运行`pip install -r requirements.txt`安装对应依赖,其中`pytorch`和`mindspore`请自行安装

在该目录下,运行`auto_dino_torch_LSTM.py`或者 `auto_dino_mindspore_LSTM.py`,按照命令行提示操作,即可直接运行

### 目录说明

1. `auto_dino`: dino最终部署运行程序
2. `ckpt`: 最终训练模型,不同模型保存文件转化脚本
3. `data`: 数据收集脚本,数据整理查看脚本,训练数据路径
4. `models`: 搭建的LSTM网络结构
5. `train_test`: 训练测试脚本
6. `utils`: 通用函数
