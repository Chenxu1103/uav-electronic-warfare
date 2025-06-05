# 用户指南：基于深度强化学习的多无人机电子对抗决策算法

本指南将帮助您成功运行和使用多无人机电子对抗决策算法实现。这是对博士论文中提出的算法的复现，特别是第3章系统模型和第5章基于动作依赖的强化学习算法（AD-PPO）。

## 第一部分：环境设置

### 系统要求

- **操作系统**：支持Windows、macOS或Linux
- **Python版本**：Python 3.7或更高版本
- **内存**：建议至少8GB RAM
- **存储空间**：至少500MB可用空间
- **GPU**：训练大型模型时推荐使用，但CPU也可以运行

### 安装步骤

1. **安装Python**：
   - 从[Python官网](https://www.python.org/downloads/)下载并安装适合您系统的Python版本
   - 确保添加Python到系统PATH

2. **克隆或下载项目**：
   - 如果使用Git：`git clone <项目仓库URL>`
   - 或直接下载项目ZIP文件并解压

3. **安装依赖**：
   - 打开终端/命令提示符，导航到项目根目录
   - 运行：`pip install -r requirements.txt`

4. **验证安装**：
   - 运行：`python check_imports.py`
   - 如果所有检查都通过，则环境设置成功

## 第二部分：项目结构

项目按以下结构组织：

```
.
├── README.md                     # 项目说明
├── requirements.txt              # 依赖环境
├── run.py                        # 项目启动脚本
├── check_imports.py              # 环境检查脚本
├── USER_GUIDE.md                 # 用户指南（本文档）
├── src                           # 源代码
│   ├── models                    # 系统模型
│   ├── algorithms                # 算法实现
│   ├── utils                     # 工具函数
│   └── main.py                   # 主程序入口
├── notebooks                     # Jupyter笔记本
│   └── evaluation.py             # 性能评估与可视化
└── experiments                   # 实验结果保存
    └── results                   # 实验结果数据
```

主要组件说明：

- **models**：实现了无人机运动学、雷达探测和环境模型
- **algorithms**：包含AD-PPO、MADDPG和ADA-RL算法实现
- **utils**：提供可视化、指标计算等工具函数
- **notebooks**：包含用于评估算法性能的脚本

## 第三部分：基本使用

### 运行方式

项目提供了统一的启动脚本`run.py`，它会自动设置正确的Python路径。所有操作都应通过此脚本执行：

```bash
python run.py [参数]
```

### 常用命令

1. **训练算法**：

   ```bash
   # 训练AD-PPO算法100次迭代
   python run.py --train --algorithms ad_ppo --ppo_iterations 100
   
   # 训练AD-PPO和MADDPG算法并比较
   python run.py --train --algorithms ad_ppo,maddpg --episodes 300
   ```

2. **评估算法**：

   ```bash
   # 评估已训练的算法
   python run.py --evaluate --algorithms ad_ppo,maddpg
   ```

3. **可视化结果**：

   ```bash
   # 生成可视化结果
   python run.py --visualize --algorithms ad_ppo,maddpg
   ```

4. **通过评估脚本进行全面分析**：

   ```bash
   # 运行评估脚本
   python notebooks/evaluation.py
   ```

### 参数说明

- `--train`: 训练模式
- `--evaluate`: 评估模式  
- `--visualize`: 可视化模式
- `--algorithms`: 要使用的算法，用逗号分隔（可选：ad_ppo, maddpg, ada_rl）
- `--episodes`: 训练回合数（针对ADA-RL和MADDPG，默认300）
- `--max_steps`: 每回合最大步数（默认200）
- `--ppo_iterations`: AD-PPO算法训练迭代次数（默认100）
- `--save_path`: 结果保存路径（默认为experiments/results）

## 第四部分：算法详解

### AD-PPO（基于动作依赖的PPO算法）

AD-PPO是本项目的核心算法，它基于PPO框架，引入了动作依赖机制以提高无人机电子对抗任务的决策效果。主要特点包括：

1. **动作依赖机制**：
   - 将无人机动作空间分为移动动作和干扰动作
   - 干扰动作依赖于移动动作，形成层次化决策结构
   
2. **网络架构**：
   - 共享特征提取层处理环境状态
   - 移动动作头生成连续的移动决策
   - 依赖层将移动动作信息融入干扰决策
   - 干扰动作头生成二值决策（是否干扰）和连续决策（干扰方向）
   
3. **PPO训练**：
   - 使用GAE计算优势函数
   - 应用PPO的截断目标函数
   - 使用熵正则化促进探索

详细的算法设计文档可以参考 `AD_PPO_DESIGN.md`。

### MADDPG（多智能体深度确定性策略梯度）

MADDPG是一种经典的多智能体强化学习算法，在本项目中作为对比算法。其特点包括：

1. **集中式训练、分布式执行**：
   - 训练时利用全局信息
   - 执行时每个智能体只使用自己的观测

2. **Actor-Critic架构**：
   - Actor网络为每个智能体生成确定性动作
   - Critic网络评估所有智能体的联合动作值

### ADA-RL（基于动作依赖的强化学习）

ADA-RL是另一种基于动作依赖的算法，基于SAC框架实现。它与AD-PPO共享动作依赖思想，但在实现上有所不同。

## 第五部分：实验结果解析

训练完成后，您将获得以下类型的结果：

1. **训练曲线**：显示奖励和损失随训练进度的变化
2. **性能指标**：包括任务完成率、探测躲避率、干扰效果等
3. **轨迹可视化**：无人机运动轨迹和雷达覆盖范围的2D/3D图

所有结果将保存在 `experiments/results` 目录下，您可以通过以下方式查看：

- 直接打开保存的图像文件
- 运行 `python notebooks/evaluation.py` 进行综合分析
- 检查保存的CSV文件了解详细指标

## 第六部分：故障排除

以下是一些常见问题及其解决方法：

### 导入错误

**问题**：无法导入模块，比如 `ModuleNotFoundError: No module named 'src.models'`

**解决方法**：
1. 确保使用 `run.py` 而不是直接运行 `src/main.py`
2. 运行 `python check_imports.py` 检查环境
3. 确保从项目根目录运行命令

### 依赖问题

**问题**：缺少依赖，比如 `ModuleNotFoundError: No module named 'gym'`

**解决方法**：
1. 运行 `pip install -r requirements.txt`
2. 对于特定的依赖问题，可以单独安装：`pip install gym`

### 形状不匹配错误

**问题**：运行时出现形状不匹配错误，如 `low.shape doesn't match provided shape`

**解决方法**：
1. 检查环境配置，特别是无人机数量和动作空间维度
2. 确认 `ECMEnvironment` 类的初始化参数设置正确

### GPU相关问题

**问题**：CUDA错误或警告

**解决方法**：
1. 对于不需要GPU的实验，可以设置：`export CUDA_VISIBLE_DEVICES=-1`
2. 更新GPU驱动程序和CUDA版本
3. 尝试使用CPU版本的PyTorch

## 第七部分：扩展和定制

您可以通过以下方式扩展或定制项目：

1. **添加新算法**：
   - 在 `src/algorithms/` 目录下创建新的算法实现
   - 更新 `src/algorithms/__init__.py` 导出新算法
   - 在 `src/main.py` 中添加对应的训练和评估代码

2. **修改环境设置**：
   - 在 `setup_environment()` 函数中调整参数
   - 修改雷达位置、无人机数量等配置

3. **自定义评估指标**：
   - 在 `src/utils/metrics.py` 中添加新的评估指标
   - 更新评估脚本以包含新指标

## 联系与支持

如果您在使用过程中遇到任何问题，请参考以下资源：

- 项目文档：README.md, AD_PPO_DESIGN.md
- 检查脚本：check_imports.py
- 原始论文：基于深度强化学习的多无人机电子对抗决策算法研究

感谢您使用本项目！希望这个实现能对您的研究或学习有所帮助。 