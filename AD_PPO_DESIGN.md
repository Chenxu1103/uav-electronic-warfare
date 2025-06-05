# 基于动作依赖的PPO算法（AD-PPO）设计文档

## 1. 算法概述

Action-Dependent PPO (AD-PPO) 是一种针对多无人机电子对抗场景设计的强化学习算法，它基于标准的Proximal Policy Optimization (PPO) 算法框架，并引入了动作依赖机制以处理无人机的复合动作空间（移动动作和干扰动作）之间的依赖关系。本算法针对论文第5章提出的基于动作依赖的强化学习算法进行设计，旨在提高多无人机在电子对抗任务中的协同决策能力。

## 2. 关键创新点

AD-PPO的主要创新点在于引入了**动作依赖机制**，使得无人机的干扰决策依赖于其移动决策。具体而言：

1. **动作分解与依赖**：将无人机的动作空间分解为移动动作子空间（加速度、转向角速度）和干扰动作子空间（是否干扰、干扰方向），并建立干扰动作对移动动作的依赖关系。
2. **层次化决策**：首先基于环境状态决定移动动作，然后基于状态和已确定的移动动作共同决定干扰动作，形成层次化决策过程。
3. **PPO框架整合**：将动作依赖机制整合到PPO算法框架中，利用PPO的稳定性和样本效率，同时通过动作依赖提高多智能体协同能力。

## 3. 算法详细设计

### 3.1 网络架构

AD-PPO的核心是一个特殊设计的**ActorCritic**网络，其架构如下：

#### 特征提取层

```python
self.feature_extractor = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU()
)
```

特征提取层负责从环境状态中提取关键特征，为后续的动作预测和价值估计提供基础。

#### 移动动作输出层

```python
self.movement_mean = nn.Linear(hidden_dim, 2)  # [accel, turn_rate]
self.movement_log_std = nn.Parameter(torch.zeros(1, 2) - 0.5)
```

移动动作输出层生成移动动作的均值和标准差，用于构建正态分布并采样连续的移动动作。

#### 动作依赖层

```python
self.dependency_layer = nn.Sequential(
    nn.Linear(hidden_dim + 2, hidden_dim),  # 特征+移动动作
    nn.ReLU()
)
```

动作依赖层是算法的核心，它将状态特征和移动动作融合，生成用于预测干扰动作的新特征表示。这种结构使得干扰决策能够考虑到移动决策的影响。

#### 干扰动作输出层

```python
self.jamming_prob = nn.Sequential(
    nn.Linear(hidden_dim, 1),
    nn.Sigmoid()
)
self.jamming_direction_mean = nn.Linear(hidden_dim, 3)  # 3D干扰方向向量
self.jamming_direction_log_std = nn.Parameter(torch.zeros(1, 3) - 0.5)
```

干扰动作输出层基于依赖层的输出生成两部分决策：
- 是否进行干扰（二值决策，使用Sigmoid函数）
- 干扰方向（连续决策，类似移动动作使用正态分布）

#### 价值函数估计

```python
self.value = nn.Linear(hidden_dim, 1)
```

价值函数网络用于估计状态价值，辅助PPO算法的更新过程。

### 3.2 动作生成流程

AD-PPO的动作生成遵循依赖关系，具体流程如下：

1. **特征提取**：首先从环境状态中提取特征表示。
2. **生成移动动作**：基于提取的特征，生成移动动作分布，并从中采样得到具体的移动动作。
3. **特征与移动动作融合**：将提取的特征与已采样的移动动作连接，输入动作依赖层。
4. **生成干扰动作**：基于融合后的特征，生成干扰决策（是否干扰）和干扰方向。
5. **组合完整动作**：将移动动作和干扰动作组合成完整的动作向量返回给环境。

关键代码实现：

```python
def act(self, state):
    with torch.no_grad():
        forward_result = self.forward(state)
        features = forward_result['features']
        
        # 采样移动动作
        movement_dist = forward_result['movement_dist']
        movement_action = movement_dist.sample()
        
        # 基于移动动作采样干扰动作
        jamming_result = self.get_dependent_jamming_action(features, movement_action)
        jamming_prob = jamming_result['jamming_prob']
        jamming_binary = torch.bernoulli(jamming_prob)  # 是否干扰
        
        # 采样干扰方向
        jamming_direction_dist = jamming_result['jamming_direction_dist']
        jamming_direction = jamming_direction_dist.sample()
        
        # 组合完整动作
        action = torch.cat([
            movement_action,
            jamming_binary,
            jamming_direction
        ], dim=-1)
        
    return action, total_log_prob, value
```

### 3.3 PPO更新流程

AD-PPO使用标准PPO的更新机制，但需要特殊处理动作依赖关系：

1. **收集经验轨迹**：使用当前策略与环境交互，收集状态、动作、奖励等数据。
2. **计算GAE优势**：使用广义优势估计方法计算每个状态-动作对的优势值。
3. **策略更新**：基于截断的PPO目标函数更新策略网络，同时考虑动作依赖关系。
4. **价值函数更新**：基于收集的数据更新价值函数网络。

关键代码实现：

```python
# 策略损失 (PPO剪切目标函数)
ratio = torch.exp(new_log_probs - batch_old_log_probs)
surr1 = ratio * batch_advantages
surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
actor_loss = -torch.min(surr1, surr2).mean()

# 价值损失
critic_loss = F.mse_loss(values, batch_returns)

# 熵损失 (鼓励探索)
entropy_loss = -entropy.mean()

# 总损失
loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss
```

### 3.4 动作概率计算

由于AD-PPO处理复合动作和依赖关系，动作概率计算变得复杂：

1. **移动动作概率**：基于正态分布计算连续移动动作的概率密度。
2. **干扰决策概率**：基于伯努利分布计算二值干扰决策的概率。
3. **干扰方向概率**：只有当干扰激活时才考虑，基于正态分布计算。
4. **总动作概率**：综合上述三部分计算总的动作概率。

关键代码实现：

```python
# 计算总的动作对数概率（用于PPO更新）
total_log_prob = movement_log_prob
total_log_prob = total_log_prob + jamming_log_prob.squeeze(-1)
# 只有当干扰激活时，才考虑方向概率
total_log_prob = total_log_prob + jamming_binary.squeeze(-1) * jamming_direction_log_prob
```

## 4. 与标准PPO的区别

AD-PPO与标准PPO算法的主要区别在于：

1. **动作空间处理**：标准PPO通常处理单一连续或离散动作空间，而AD-PPO处理复合动作空间并建立其中的依赖关系。
2. **策略网络设计**：AD-PPO使用层次化结构的策略网络，将部分动作的决策依赖于其他动作的决策结果。
3. **动作概率计算**：AD-PPO需要特殊处理复合动作和条件依赖关系下的动作概率计算。
4. **多智能体协同**：AD-PPO的动作依赖机制更适合多智能体协同场景，能够在考虑个体移动决策的基础上做出更优的干扰决策。

## 5. 与MADDPG的比较

AD-PPO与MADDPG算法在多智能体学习方面有以下关键区别：

1. **策略类型**：MADDPG使用确定性策略，而AD-PPO使用随机策略，具有更好的探索能力。
2. **样本效率**：PPO基础的AD-PPO通常比基于经验回放的MADDPG具有更高的样本效率。
3. **动作依赖**：AD-PPO特别设计了动作间的依赖关系，而MADDPG没有显式处理动作空间内部的依赖。
4. **算法稳定性**：PPO的截断目标函数使AD-PPO训练更加稳定，不易出现Q值过高估计等问题。

## 6. 实现注意事项

在实现AD-PPO算法时，需要注意以下几点：

1. **网络初始化**：动作依赖层的初始化需要谨慎，确保初始状态下移动动作和干扰动作之间有合理的依赖关系。
2. **PPO超参数**：需要调整PPO的关键超参数，如剪切范围、熵系数、学习率等，以适应动作依赖场景。
3. **批次数据处理**：在计算复合动作的对数概率和熵时，需要正确处理条件依赖关系。
4. **奖励设计**：针对移动和干扰的协同行为设计合适的奖励函数，可能需要平衡两种动作子空间的学习速度。

## 7. 算法局限性与未来改进

尽管AD-PPO在多无人机电子对抗场景表现出色，但仍存在一些局限性：

1. **计算复杂度**：由于需要处理动作依赖关系，AD-PPO的计算复杂度高于标准PPO，尤其在智能体数量较多时。
2. **依赖结构固定**：当前实现中，依赖关系是固定的（干扰依赖移动），未来可探索更灵活的依赖结构。
3. **与环境耦合**：算法设计与具体的电子对抗环境耦合较紧，迁移到其他领域可能需要重新设计动作依赖结构。

未来可能的改进方向：

1. **自适应依赖结构**：设计能够自动学习最优依赖结构的机制，不需要人为指定依赖关系。
2. **多层次依赖**：扩展到多层次的动作依赖，处理更复杂的决策场景。
3. **通信机制整合**：将智能体间的通信机制与动作依赖机制相结合，进一步提升协同效果。
4. **迁移学习**：设计能够在不同环境间迁移的动作依赖模式，提高算法的通用性。

## 8. 参考文献

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
2. Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems (pp. 6379-6390).
3. Chen, J., Wang, Z., Dong, M., Wang, Y., & Chen, T. (2023). Multiple-UAV Reinforcement Learning Algorithm Based on Improved PPO in Ray Framework. Drones, 7(6), 166. 