# 快速开始指南

本指南将帮助您在个人笔记本电脑上快速运行多无人机电子对抗决策算法项目。

## 环境要求

### 最低硬件要求
- **CPU**: Intel i5 或 AMD Ryzen 5 及以上
- **内存**: 8GB RAM（推荐16GB）
- **存储**: 5GB可用空间
- **GPU**: 可选，支持CUDA的显卡可以加速训练

### 软件要求
- **操作系统**: Windows 10/11, macOS 10.14+, 或 Linux Ubuntu 18.04+
- **Python**: 3.7-3.9（推荐3.8）

## 一键安装和运行

### 步骤1: 环境设置

#### Windows用户
```bash
# 1. 下载并安装Python 3.8
# 从 https://www.python.org/ 下载Python 3.8

# 2. 打开命令提示符(cmd)或PowerShell
# 3. 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 4. 升级pip
python -m pip install --upgrade pip
```

#### macOS用户
```bash
# 1. 安装Homebrew（如果没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装Python
brew install python@3.8

# 3. 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate

# 4. 升级pip
pip install --upgrade pip
```

#### Linux (Ubuntu)用户
```bash
# 1. 更新系统并安装Python
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev python3-pip

# 2. 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate

# 3. 升级pip
pip install --upgrade pip
```

### 步骤2: 安装项目依赖

```bash
# 安装所有依赖包
pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1
pip install numpy==1.21.0
pip install matplotlib==3.3.4
pip install seaborn==0.11.1
pip install pandas==1.3.0
pip install gym==0.18.3
pip install tqdm==4.61.2
pip install scikit-optimize==0.8.1
```

如果遇到安装问题，可以使用以下替代方案：
```bash
# 使用清华源镜像加速下载（中国用户推荐）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.8.1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy matplotlib seaborn pandas gym tqdm scikit-optimize
```

### 步骤3: 验证安装

运行以下命令验证安装是否成功：

```bash
python -c "import torch; import numpy; import matplotlib; print('✅ 所有依赖安装成功！')"
```

## 快速测试运行

### 1. 生成可视化演示（30秒）

```bash
# 生成态势图、轨迹图和算法对比图
python src/utils/run_visualization.py --all
```

这将在 `experiments/visualization/output/` 目录下生成三个演示图表。

### 2. 快速算法对比（2-3分钟）

```bash
# 运行短时间的算法对比测试
python compare_algorithms.py --num_episodes 5 --eval_interval 2 --eval_episodes 2
```

### 3. 完整训练实验（15-30分钟）

```bash
# 运行完整的算法对比实验
python compare_algorithms.py --num_episodes 100 --eval_interval 10 --eval_episodes 3
```

## 主要功能演示

### 功能1: AD-PPO与MADDPG性能对比

```bash
# 基本对比（推荐新手）
python compare_algorithms.py --num_episodes 50 --eval_interval 10

# 完整对比（推荐研究使用）
python compare_algorithms.py --num_episodes 500 --eval_interval 50 --hidden_dim 256
```

**结果位置**: `experiments/algorithm_comparison/时间戳/comparison/`
- `algorithm_comparison.png`: 性能对比图表
- `performance_comparison.csv`: 数值对比表格

### 功能2: 自动参数调优

```bash
# 自动寻找AD-PPO的最佳参数
python autotuner.py --algorithm ad_ppo --method grid --num_episodes 20 --quick

# 使用优化后的参数运行实验
python run_autotuned.py --algo both --num_episodes 100 --latest
```

### 功能3: 课程学习训练

```bash
# 通过逐步增加难度来训练智能体
python run_curriculum.py --algorithm ad_ppo --episodes_per_stage 50 --eval_episodes 3
```

### 功能4: 模型评估和可视化

```bash
# 评估训练好的模型
python run_eval.py --model_path experiments/results/ad_ppo/model_final.pt --algorithm ad_ppo --num_episodes 5

# 生成所有类型的可视化图表
python src/utils/run_visualization.py --all
```

## 结果解读

### 性能指标说明

- **平均奖励**: 算法在环境中获得的平均奖励值，越高越好
- **成功率**: 完成电子对抗任务的成功百分比，目标是100%
- **干扰率**: 成功干扰雷达的百分比，反映算法的核心能力
- **训练稳定性**: 通过损失曲线的平滑程度来判断

### 图表解读

1. **态势图**: 显示无人机和雷达的位置关系，干扰连线表示正在进行的电子对抗
2. **轨迹图**: 展示无人机的飞行路径，不同颜色区分不同无人机
3. **对比图**: 对比不同算法在各项指标上的表现，包含置信区间

## 常见问题解决

### Q1: 安装torch时出现错误
**解决方案**:
```bash
# 使用CPU版本的PyTorch
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Q2: 运行时出现"CUDA out of memory"错误
**解决方案**:
```bash
# 使用更小的参数
python compare_algorithms.py --num_episodes 50 --hidden_dim 128 --batch_size 32
```

### Q3: 训练速度很慢
**解决方案**:
- 使用快速模式: `--quick`
- 减少网络层数: `--hidden_dim 128`
- 减少评估频率: `--eval_interval 20`

### Q4: 内存不足
**解决方案**:
```bash
# 使用更小的批次大小和缓冲区
python compare_algorithms.py --batch_size 16 --buffer_size 50000
```

### Q5: 图表显示异常
**解决方案**:
```bash
# 重新生成可视化
python src/utils/run_visualization.py --all --output_dir new_output
```

## 获取帮助

### 查看命令行参数
```bash
# 查看算法对比的所有参数
python compare_algorithms.py --help

# 查看可视化工具的所有参数
python src/utils/run_visualization.py --help

# 查看自动调优的所有参数
python autotuner.py --help
```

### 项目结构说明

```
项目根目录/
├── src/                      # 核心源代码
│   ├── algorithms/           # 算法实现（AD-PPO, MADDPG）
│   ├── models/              # 模型实现（UAV, 雷达）
│   ├── environment/         # 环境实现
│   └── utils/               # 工具函数（可视化等）
├── experiments/             # 实验结果存储
├── compare_algorithms.py    # 主要的算法对比脚本
├── autotuner.py            # 自动参数调优脚本
├── run_curriculum.py       # 课程学习脚本
└── requirements.txt        # 依赖列表
```

## 推荐的完整实验流程

1. **验证安装** (1分钟)
```bash
python src/utils/run_visualization.py --all
```

2. **快速测试** (3分钟)
```bash
python compare_algorithms.py --num_episodes 10 --eval_interval 5 --eval_episodes 2
```

3. **参数优化** (10分钟)
```bash
python autotuner.py --algorithm ad_ppo --method grid --num_episodes 20 --quick
```

4. **完整实验** (30分钟)
```bash
python run_autotuned.py --algo both --num_episodes 200 --eval_interval 20 --latest
```

5. **课程学习** (可选，20分钟)
```bash
python run_curriculum.py --algorithm ad_ppo --episodes_per_stage 50
```

6. **结果分析**
查看 `experiments/` 目录下的所有生成的图表和数据文件。

完成以上步骤后，您将获得完整的算法对比结果，包括性能图表、数据表格和可视化展示。 