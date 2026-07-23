# FlashShot 产品级架构验收评估（2026-07-02）

> ⚠️ **本文件为 2026-07-02 时点的评估快照,保留作历史记录。** 此后已发生重大变化:
> - **PolicyEngine / LearningLayer / SmartModelRouter** 已落地为运行时活跃路径(决策从纯规则状态机升级为自适应);
> - **SiliconFlow 成为默认生成后端**(Qwen-Image-Edit + Qwen2.5-VL judge),Chrome 退为 legacy 可选项;
> - 后端测试数从当时的规模增至 **241**,commits 至 **13**。
>
> **当前结构状态以根目录 [`CLAUDE.md`](CLAUDE.md) 为准;本评估下文不再逐条订正。**

> 评估视角：产品级架构验收
> 评估对象：6 commits + 完整代码结构
> 结论：从 "MVP 写真生成器" → "模块化 AI 写真系统 v1.0"

---

## 一、整体评价

| 层 | 状态 | 说明 |
|----|------|------|
| 产品闭环 | A | 上传→生成→预览→付费→交付 完整可用 |
| Aha Moment（Hero Preview） | A | 4 候选生成+自动筛选+只展示 1 张，已系统化 |
| 支付系统 | A | Paddle webhook + 前后端一致 |
| 生成 Pipeline | B+ | 候选→评判→修复→选择，但仍是静态规则 |
| Agent Architecture | A- | Generator / Evaluator / Router 已分离 |
| 模块解耦 | A- | evaluation / generation / repair / delivery 已独立 |
| 可扩展性 | A- | Provider 抽象支持未来多模型 |
| 研究性耦合风险 | ⚠️ 已显著下降 | gemini_worker 从 2820→1905 行 |

**核心结论：**

> 系统已完成 "AI pipeline → AI product system" 转换的 **70%**。

---

## 二、6 个 Commits 的架构意义

### Commit 1：Hero Preview = 产品核心成立

**本质不是：** "多生成 4 张图"

**本质是：** 把 Aha Moment 从**偶然事件** → **系统保证行为**

- 以前：靠单次模型随机性
- 现在：可控曝光 + ranking + 只展示最佳

**意义：** 转化率系统化的第一步。

---

### Commit 2+3：Evaluation + Router 拆分

**本质升级：**

```
以前：procedural pipeline（过程式脚本）
以后：policy-driven system（策略驱动系统）
```

现在结构：
```
Generator（生成）
Evaluator（判断）
Router（决策）
```

**这是 AI 产品和 AI demo 的分水岭。**

⚠️ **但当前 Router 仍是 rule-based deterministic policy：**
- 无法学习
- 无法优化
- 无法自动提升 identity pass rate

**下一步必须：** `rule → hybrid policy → learned ranking`

---

### Commit 4：Gateway + Provider 抽象

**本质：** 解决了 model lock-in problem（模型锁定风险）

现在结构：
```
ImageProvider
   ├── OpenRouterProvider
   ├── ChromeProvider (legacy)
   └── Local repair
```

⚠️ **但当前 gateway 是 "thin wrapper"，不是 "smart router"：**
- 没有 routing strategy
- 没有 cost-aware selection
- 没有 quality-aware selection

**下一步必须：** `model selection policy = f(cost, identity, aesthetics, latency)`

---

### Commit 5：Multi-style 强制 Aha

**产品判断非常成熟。**

修复了关键问题：bundle 用户没有 Aha moment。

核心逻辑正确：
```
multi-style → still goes through hero preview
```

⚠️ **但缺少 identity-conditioned style ranking：**
- 同一人在不同风格下的一致性评分未建立
- 未来需要 `same person → different style → consistency scoring`

---

### Commit 6：Overview 文档

**System crystallization step（系统结晶化）。**

不是代码贡献，是：
- 产品定义固定
- Pipeline 固定
- Team onboarding ready

✔ 很多团队缺这个。

---

## 三、当前系统真实状态

```
User Input
   ↓
Hero Preview System  ← 产品核心
   ↓
Evaluation Layer     ← 质量系统（静态）
   ↓
Router Layer         ← 决策系统（静态规则）
   ↓
Generation Engine    ← 模型层（单模型）
   ↓
Postprocess + Delivery
```

---

## 四、当前最大问题（只说一个）

### ❗ 系统还不是 "learning system"

现在是：
```
rule-based decision system
```

而不是：
```
adaptive optimization system
```

**表现为：**
- threshold fixed（阈值固定）
- router static（路由静态）
- evaluation static（评估静态）
- no feedback loop（无反馈闭环）

**结果：**
> identity pass rate improvement 上限被锁死

---

## 五、下一阶段（真正产品化关键）

### Step 1：Evaluation → Learning Layer

现在：
```
score + rule
```

未来：
```
score + distribution + calibration
```

**要做：**
- user "not like me" → hard negative label
- user "selected" / "downloaded" → positive reinforcement
- 反馈影响未来 threshold 和 shot selection

---

### Step 2：Router → Policy Engine

现在：
```
if identity < threshold → regenerate
```

未来：
```
policy(state) → action probability
```

| before | after |
|--------|-------|
| rule | policy |
| deterministic | probabilistic |
| static | adaptive |

---

### Step 3：Model Gateway → Smart Router

现在：
```
just abstraction
```

未来：
```
cost × identity × aesthetics × latency → routing decision
```

**任务级模型选择：**
| 任务 | 模型 |
|------|------|
| Hero face | identity-stable model |
| Full body | composition model |
| Local edit | inpainting model |
| Upscale | RealESRGAN |

---

## 六、系统真实定位

| 阶段 | 标签 |
|------|------|
| 当前 | **AI Portrait Production System v1** |
| 下一阶段 | **AI Identity-Constrained Generative Platform** |
| 最终形态 | **AI Personal Visual Identity Engine** |

---

## 七、最关键建议（只做 3 件事）

### 1️⃣ 把 Evaluation 变成"可学习系统"

不是 score，是 **feedback loop**。

### 2️⃣ 把 Router 变成 Policy System

不是 if/else，是 **decision engine**。

### 3️⃣ 把 Hero Preview 变成"唯一增长核心"

所有优化围绕：
```
Hero → Pay conversion rate
```

---

## 八、一句话总结

> 这个系统已经不是"AI 写真项目"了。
> 
> 你已经在做一个 **"身份约束生成系统（Identity-Constrained Generative System）"**。

---

## 附录：当前代码健康度

| 指标 | 值 |
|------|-----|
| 后端测试 | 136/136 通过 |
| 前端 build | 通过 |
| gemini_worker.py | 1905 行（从 2820 行降下） |
| 新增模块 | generation/, evaluation/, repair/, delivery/ |
| Git commits | 6 个 |
| 代码压缩包 | 321KB（已排除图片/模型/数据） |

**结论：架构债务已显著降低，系统进入"可演进"状态。**
