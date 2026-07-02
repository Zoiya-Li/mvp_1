# A/B 实验报告：Generation vs. Editing 身份保持写真

## 目标

验证「图像编辑（image editing）」是否比「文生图生成（text-to-image generation）」更适合 FlashShot 的 identity-preserving 写真，并比较 OpenRouter 上多个 image-generation 模型的表现。

## 方法

- **测试脚本**：`experiments/compare_identity_preservation.py`
- **数据集**：2 位男性真实自拍主体 × 3 套风格模板（共 6 个 case）
  - `s_8ccf_iphone`：iPhone 人像模式自拍
  - `s_b735`：上传的正面半身照
  - 模板：`gf_m_hanfu`（国风汉服）、`bf_m_04`（现代商务）、`film_m_cyber`（电影感）
- **评估**：生产环境同款 strict Chinese judge（`RESEMBLANCE_JUDGE_PROMPT`），满分 10 分，≥8 视为「明显同一人」。
- **对比维度**：
  - **Framing**：`generation`（模板在前，用户在后） vs. `editing`（用户在前，模板在后，显式要求「保持人脸，只换衣服/背景」）。
  - **Models**：Gemini 3.1 Flash / Gemini 3 Pro / Gemini 2.5 Flash / GPT-5 Image / GPT-5 Image-mini。

## 核心结果

| 模型 | Framing | 平均相似分 | pass≥8 比例 | 备注 |
|---|---|---|---|---|
| google/gemini-3.1-flash-image-preview | **editing** | **6.50** | **33%** | 当前最优，速度 ~20–40s |
| google/gemini-3-pro-image-preview | editing | 6.17 | 17% | 质量不错但贵且慢 |
| google/gemini-3.1-flash-image-preview | generation | 5.33 | 0% | 当前生产 baseline |
| google/gemini-3-pro-image-preview | generation | 4.67 | 0% | 比 Flash 还差 |
| google/gemini-2.5-flash-image | both | 5.00 | 0% | 不推荐 |
| openai/gpt-5-image-mini | editing | 5.50 | 0% | 慢 ~60–80s |
| openai/gpt-5-image | editing | ~5.0 | 0% | 极慢（一次 21 分钟），一次失败 |

> 多图编辑（2–3 张 selfie）未带来明显提升：s_8ccf 仍为 6 分，s_b735 仍为 9 分。

## 关键发现

1. **Editing 显著优于 Generation**  
   在 Gemini 系列上，editing framing 的平均分比 generation 高 **1–1.5 分**，且是唯一能冲击 8–9 分的方案。

2. **Gemini 3.1 Flash 是当前最优单模型**  
   它在 editing 模式下拿到 2 个 9/10、1 个 8/10（潜在），平均分最高， latency 最低，成本最低。

3. **Gemini 3 Pro 不是全面升级**  
   Pro 在 generation 上反而更差，editing 上略逊 Flash，且更贵更慢。不需要为身份保持而升级模型。

4. **OpenAI GPT-5 Image 不适合生产**  
   GPT-5 Image-mini 和 GPT-5 Image 的 editing 均只能到 5–6 分；GPT-5 Image 单次耗时可达 21 分钟，不可接受。

5. **身份保持仍高度依赖用户自拍质量**  
   同一模型对 `s_b735` 可稳定 8–9 分，对 `s_8ccf_iphone` 却很难突破 6 分。说明自拍角度/光线/表情仍是上限因素，未来可考虑上传引导或预处理。

## 生产决策

**结论**：FlashShot 应升级为「单模型 Editing Pipeline」，而非多模型混合方案。

- **采用**：`google/gemini-3.1-flash-image-preview` + editing prompt
- **不采用**：Gemini 3 Pro、GPT-5 Image、FLUX/Riverflow（OpenRouter 当前未在 models API 中列出可用 image output）
- **后续可探索**：如果 editing pipeline 仍无法解决某些 hard case，再考虑 VTO（虚拟试衣）+ scene fusion 的两阶段混合方案。

## 已落地的改动

- `server/openrouter_client.py`：新增 `editing_mode` 参数，支持 `[user_selfies..., template]` 的图片顺序。
- `server/gemini_worker.py`：新增 `build_editing_prompt()`，初始生成默认使用 editing framing。
- `server/job_queue.py`：向 worker 传入 `gen_prompt`（纯风格描述），避免与 editing 的图序冲突。
- `tests/test_resemblance_loop.py`：更新 FakeClient 签名，测试通过。

## 验证状态

- `pytest tests/`：36 passed
- `npm run build`（frontend）：通过
- 本地 dev server：backend `:8000`、frontend `:3001` 均已重启并 200

## 附件

- 原始输出：`experiments/output/*.png`
- 可复现实验：`experiments/compare_identity_preservation.py`
- 报告数据：`experiments/output/report.json` / `report.csv`
