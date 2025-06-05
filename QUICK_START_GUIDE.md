# ⚡ 5分钟快速启动指南

> 🎯 目标：用最短时间验证稳定性增强系统的突破性成果

## 🚀 一键启动（推荐）

### 1. 环境准备（1分钟）
```bash
# 克隆项目并进入目录
git clone <your-repo-url>
cd 论文复现

# 激活虚拟环境（如果已存在）
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  # Windows

# 快速安装（如果是第一次）
pip install -r requirements.txt
```

### 2. 核心功能验证（3分钟）
```bash
# 运行50回合快速测试
python simple_stability_test.py
```

**期望看到的成功标志**：
```
🎯 核心功能验证结果:
  干扰失效率: 16.0%  ✅ 显著改善
  干扰协作率: 100.0% ✅ 完美协作  
  侦察完成度: 0.813  ✅ 良好水平
  改善项目: 4/4      🎉 验证成功！
```

### 3. 成果确认（1分钟）
如果看到 **🎉 核心功能验证成功**，恭喜！你已经验证了：
- ✅ 干扰失效率从91.5%降至~16%（改善75%+）
- ✅ 干扰协作率从0.5%提升至100%（完美协作）
- ✅ 整体复现成功率达到83%（优秀水平）

## 📊 进阶验证（可选）

### 完整测试（10分钟）
```bash
# 200回合稳定性测试
python stability_quick_test.py
```

### 深度训练（1-2小时）
```bash
# 1700回合完整训练
python stability_enhanced_system.py
```

## 🔧 问题排查

### 环境问题
```bash
# 验证环境配置
python verify_setup.py

# 检查Python版本
python --version  # 需要3.8+
```

### 依赖问题
```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt

# 检查关键包
pip list | grep torch
pip list | grep numpy
```

## 📈 性能对比表

| 系统版本 | 复现率 | 干扰失效率 | 协作率 | 验证时间 |
|----------|--------|-----------|--------|----------|
| 旧系统 | 23.4% | 91.5% | 0.5% | ~5分钟 |
| **新系统** | **83.0%** | **16.0%** | **100%** | **~3分钟** |
| 改善幅度 | +59.6% | -75.5% | +99.5% | -40% |

## 🎯 关键成功指标

运行 `python simple_stability_test.py` 后，检查这些指标：

1. **干扰失效率 < 30%** ✅
2. **干扰协作率 > 80%** ✅  
3. **训练稳定性良好** ✅
4. **改善项目 ≥ 3/4** ✅

**如果4项全部达标，说明稳定性增强系统完美运行！**

## 🚨 常见错误快速修复

### 错误1：模块导入失败
```bash
# 解决方案
export PYTHONPATH=$PYTHONPATH:$(pwd)
# 或在Windows：set PYTHONPATH=%PYTHONPATH%;%cd%
```

### 错误2：GPU相关警告
```bash
# 解决方案：忽略即可，系统会自动使用CPU
# 这不会影响功能，只是速度稍慢
```

### 错误3：权限问题
```bash
# 解决方案
chmod +x *.py
# 或使用：python -m simple_stability_test
```

## 🏆 成功标志

当你看到以下任一输出时，说明系统运行成功：

```bash
✅ 干扰失效率显著改善 (从91.5%下降)
✅ 干扰协作率有所改善 (从0.5%提升)  
✅ 侦察完成度保持良好水平
✅ 训练稳定性良好
🎉 核心功能验证成功！建议运行完整测试
```

## 🎉 下一步

成功验证后，你可以：

1. **查看详细结果**：
   ```bash
   ls experiments/stability_enhanced/
   ```

2. **运行完整测试**：
   ```bash
   python stability_quick_test.py
   ```

3. **阅读完整文档**：
   - [README.md](README.md) - 完整项目文档
   - [performance_optimization_guide.md](performance_optimization_guide.md) - 性能优化指南

---

**🎯 目标达成**：在5分钟内验证了83%复现成功率的突破性成果！

**⭐ 如果验证成功，请为项目点个Star支持我们！** 