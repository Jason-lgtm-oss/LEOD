# LEOD论文与代码深度映射分析

## 目录

1. [论文核心内容总结](#1-论文核心内容总结)
2. [方法与代码映射](#2-方法与代码映射)
3. [关键技术深度分析](#3-关键技术深度分析)
4. [实验配置与复现指南](#4-实验配置与复现指南)
5. [代码质量与创新亮点](#5-代码质量与创新亮点)

---

## 1. 论文核心内容总结

### 1.1 研究背景与问题

**论文标题**: LEOD: Label-Efficient Object Detection for Event Cameras  
**发表会议**: CVPR 2024  
**作者**: Ziyi Wu, Mathias Gehrig, Qing Lyu, Xudong Liu, Igor Gilitschenski  

**核心问题**:
- 事件相机（Event Camera）是受生物启发的低延迟传感器，具有高时间分辨率（>1000 FPS）
- 现有数据集标注频率很低（例如4 FPS），导致大量事件数据未被标注
- 模型仅在稀疏标注帧上训练，导致性能次优、收敛速度慢

**关键洞察**:
事件数据的高时间分辨率特性与低标注频率之间存在巨大差距，可以通过弱监督/半监督学习来利用未标注的事件数据。

### 1.2 主要创新点

#### 1) 标签高效的自训练框架
- **Teacher-Student架构**: 采用Self-Training范式，教师模型在标注数据上训练后生成伪标签
- **迭代式训练**: 学生模型在伪标签+真实标签上训练，然后作为新的教师模型继续生成更高质量的伪标签
- **适配事件数据**: 针对事件相机的时间连续性特点设计伪标签生成策略

#### 2) 高质量伪标签生成
- **Test-Time Augmentation (TTA)**: 使用水平翻转和时间翻转增强来提高检测鲁棒性
- **置信度过滤**: 结合objectness和class confidence双重阈值过滤低质量预测
- **时空轨迹过滤**: 利用线性跟踪器过滤短轨迹检测，减少假阳性

#### 3) 循环时间建模
- **Recurrent Backbone**: 使用MaxViT + ConvLSTM的组合维护时间上下文
- **状态管理**: 为每个数据加载器worker维护独立的LSTM隐藏状态
- **混合数据加载**: 同时使用随机访问和流式加载来平衡探索和利用

### 1.3 实验结果与性能提升

**在Gen1数据集上（仅1%标注数据）**:
- Baseline (监督学习): ~26% mAP
- LEOD (1轮自训练): ~32% mAP (+6%)
- LEOD (2轮自训练): ~34% mAP (+8%)

**在Gen4数据集上（1Mpx，1%标注数据）**:
- Baseline: ~28% mAP
- LEOD (1轮): ~35% mAP (+7%)
- LEOD (2轮): ~37% mAP (+9%)

**关键发现**:
- 伪标签的精度（Precision）与下一轮自训练的性能提升高度相关
- 通常2轮自训练后性能增益趋于饱和
- 弱监督（WSOD）和半监督（SSOD）设置下都能显著提升性能

---

## 2. 方法与代码映射

### 2.1 整体架构映射

```
论文方法                          代码实现
┌─────────────────┐              ┌────────────────────┐
│  RVT Backbone   │    ═════>    │ RNNDetector        │
│  (Recurrent)    │              │ maxvit_rnn.py      │
└─────────────────┘              └────────────────────┘
         ↓                                 ↓
┌─────────────────┐              ┌────────────────────┐
│  Feature Neck   │    ═════>    │ PAFPN              │
│     (PAFPN)     │              │ yolox/models/pafpn │
└─────────────────┘              └────────────────────┘
         ↓                                 ↓
┌─────────────────┐              ┌────────────────────┐
│  Detection Head │    ═════>    │ YOLOXHead          │
│     (YOLOX)     │              │ yolox/yolo_head.py │
└─────────────────┘              └────────────────────┘
```

### 2.2 伪标签生成策略映射

#### 论文方法
论文第3.2节描述了伪标签生成流程：
1. 使用训练好的教师模型对所有训练序列进行推理
2. 应用TTA（水平翻转+时间翻转）提高预测质量
3. 使用双阈值过滤低置信度预测
4. 应用基于跟踪的后处理来过滤噪声

#### 代码实现：`modules/pseudo_labeler.py`

**核心类结构**:
```python
class PseudoLabeler(Module):
    """伪标签生成器，继承自检测模块"""
    
    # 关键配置
    - use_gt: 是否在标注帧使用GT标签（Gen1为True，Gen4可能False）
    - tta_cfg: TTA配置（enable, hflip, tflip）
    - obj_thresh, cls_thresh: 置信度阈值
    - min_track_len: 最小轨迹长度
```

**TTA实现** (L467-492):
```python
# 水平翻转TTA
if self.tta_cfg.enable and self.tta_cfg.hflip:
    hflip_ev_repr = th.flip(ev_repr, dims=[-1])
    ev_repr = th.cat([ev_repr, hflip_ev_repr], dim=1)
    # 同时翻转标签
    for i, (lbl, lbl_flip) in enumerate(zip(labels, labels_flip)):
        lbl_flip.flip_lr_()
        labels[i] = lbl + lbl_flip
```

**置信度过滤** (L37-91, `tta_postprocess`函数):
```python
def tta_postprocess(preds, conf_thre=0.7, nms_thre=0.45):
    """对TTA预测应用NMS和置信度过滤"""
    obj_conf, class_conf = pred[:, 4], pred[:, 5]
    conf_mask = ((obj_conf * class_conf) >= conf_thre)  # 双重置信度
    detections = detections[conf_mask]
    # 应用NMS
    nms_out_index = ops.batched_nms(...)
```

**跟踪过滤** (L202-266, `_track`方法):
```python
def _track(labels, frame_idx, min_track_len=6, inpaint=False):
    """使用线性跟踪器过滤短轨迹"""
    model = LinearTracker(img_hw=labels[0].input_size_hw)
    # 对每一帧执行跟踪
    for f_idx in range(max(frame_idx) + 1):
        model.update(frame_idx=f_idx, dets=bboxes, is_gt=is_gt)
    # 过滤hits < min_track_len的轨迹
    if tracker.hits >= min_track_len:
        keep_bbox
    else:
        remove_bbox
```

**映射总结**:
| 论文描述 | 代码位置 | 关键参数 |
|---------|---------|---------|
| TTA增强 | `pseudo_labeler.py:L467-492` | `tta.enable`, `tta.hflip` |
| 置信度过滤 | `pseudo_labeler.py:L37-91` | `obj_thresh`, `cls_thresh` |
| 轨迹过滤 | `pseudo_labeler.py:L202-266` | `min_track_len=6` |
| 轨迹填充 | `pseudo_labeler.py:L240-265` | `inpaint=True` |

### 2.3 循环特征提取映射

#### 论文方法
论文基于RVT (Recurrent Vision Transformer)，使用ConvLSTM维护时间上下文。

#### 代码实现：`models/detection/recurrent_backbone/maxvit_rnn.py`

**RNNDetector架构** (L23-115):
```python
class RNNDetector(BaseDetector):
    """RNN检测器主干网络，使用MaxViT blocks"""
    
    def __init__(self, mdl_config):
        # 4个stage，每个stage包含MaxViT attention + ConvLSTM
        for stage_idx in range(4):
            stage = RNNDetectorStage(
                dim_in=input_dim,
                stage_dim=stage_dim,
                spatial_downsample_factor=downsample_factor,
                num_blocks=num_blocks,
                T_max_chrono_init=T_max_chrono_init_stage
            )
            self.stages.append(stage)
```

**每个Stage的前向传播** (L182-201):
```python
def forward(self, x, h_and_c_previous=None, token_mask=None):
    """单stage的RNN前向传播"""
    # 1. 空间下采样 + channel first -> last
    x = self.downsample_cf2cl(x)  # [B,C,H,W] -> [B,H,W,C]
    
    # 2. MaxViT attention (window + grid)
    for blk in self.att_blocks:
        x = blk(x)  # 保持 [B,H,W,C] 格式
    
    # 3. 转回 channel-first
    x = nhwC_2_nChw(x)  # [B,H,W,C] -> [B,C,H,W]
    
    # 4. ConvLSTM更新
    h_c_tuple = self.lstm(x, h_and_c_previous)
    x = h_c_tuple[0]  # 使用hidden state作为输出
    
    return x, h_c_tuple  # 返回特征和LSTM状态
```

**LSTM状态管理**：`modules/detection.py:L170-228`
```python
# 为每个dataloader worker维护独立的RNN状态
self.mode_2_rnn_states[mode].reset(
    worker_id=worker_id, 
    indices_or_bool_tensor=is_first_sample
)

# 获取上一时刻的状态
prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)

# 前向传播
backbone_features, states = self.mdl.forward_backbone(
    x=ev_tensors,
    previous_states=prev_states
)

# 保存并detach状态（防止梯度累积）
self.mode_2_rnn_states[mode].save_states_and_detach(
    worker_id=worker_id, 
    states=prev_states
)
```

**映射总结**:
| 论文组件 | 代码实现 | 位置 |
|---------|---------|------|
| Recurrent Backbone | RNNDetector | `maxvit_rnn.py:L23-115` |
| MaxViT Block | MaxVitAttentionPairCl | `maxvit_rnn.py:L118-139` |
| ConvLSTM | DWSConvLSTM2d | `models/layers/rnn.py` |
| 状态管理 | RNNStates | `detection.py:L51-55` |
| Per-worker状态 | mode_2_rnn_states | `detection.py:L170-228` |

### 2.4 数据混合策略映射

#### 论文方法
论文第4.3节提到使用混合训练策略：同时使用随机采样和流式采样。

#### 代码实现：`modules/data/genx.py`

**混合采样配置** (L120-144):
```python
def set_mixed_sampling_mode_variables_for_train(self):
    """根据权重分配random和stream的batch size和workers"""
    weight_random = self.dataset_config.train.mixed.w_random
    weight_stream = self.dataset_config.train.mixed.w_stream
    
    # 按权重分配batch size
    bs_rnd = round(total_bs * weight_random / (weight_stream + weight_random))
    bs_str = total_bs - bs_rnd
    
    # 按batch size分配workers (random采样通常更慢)
    workers_rnd = ceil(total_workers * bs_rnd / total_bs)
    workers_str = total_workers - workers_rnd
```

**Dataloader创建** (L189-205):
```python
def train_dataloader(self):
    train_loaders = dict()
    # 为每种采样模式创建dataloader
    for sampling_mode, dataset in self.sampling_mode_2_dataset.items():
        train_loaders[sampling_mode] = DataLoader(
            dataset=dataset,
            batch_size=self.sampling_mode_2_train_batch_size[sampling_mode],
            num_workers=self.sampling_mode_2_train_workers[sampling_mode],
            ...
        )
    # 返回包含两个dataloader的字典
    return train_loaders  # {RANDOM: loader1, STREAM: loader2}
```

**训练时的batch合并**：`modules/detection.py:L150-155`
```python
def training_step(self, batch, batch_idx):
    # batch是一个dict: {RANDOM: batch1, STREAM: batch2}
    batch = merge_mixed_batches(batch)  # 合并成单个batch
    data = self.get_data_from_batch(batch)
    worker_id = self.get_worker_id_from_batch(batch)
    # 每个worker维护独立的LSTM状态
    ...
```

**映射总结**:
| 论文描述 | 代码实现 | 关键逻辑 |
|---------|---------|---------|
| 混合采样 | `genx.py:L120-144` | 按权重分配BS和workers |
| Random采样 | `dataset_rnd.py` | 随机访问标注帧 |
| Stream采样 | `dataset_streaming.py` | 顺序加载事件序列 |
| Batch合并 | `merge_mixed_batches` | 合并来自不同采样模式的batch |

### 2.5 Prophesee评估指标映射

#### 论文方法
使用Prophesee benchmark的标准mAP评估指标。

#### 代码实现：`utils/evaluation/prophesee/evaluator.py`

**评估器集成**：`modules/detection.py:L76-84`
```python
# 为每个模式创建评估器
self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
    dataset=dataset_name,
    downsample_by_2=dst_cfg.downsample_by_factor_2
)
```

**添加预测和标签**：`modules/detection.py:L291-296`
```python
if mode in self.mode_2_psee_evaluator:
    # 添加GT标签
    self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
    # 添加模型预测
    self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
    # 定期评估
    if step % self.train_eval_every == 0:
        self.run_psee_evaluator(mode=mode)
```

**运行评估**：`modules/detection.py:L409-463`
```python
def run_psee_evaluator(self, mode, log=True):
    """计算并记录Prophesee mAP指标"""
    metrics = psee_evaluator.evaluate_buffer(
        img_height=hw_tuple[0],
        img_width=hw_tuple[1]
    )
    # metrics包含: mAP, AP@0.5, AP@0.75, etc.
    self.log_dict(metrics)
```

---

## 3. 关键技术深度分析

### 3.1 伪标签生成流程详解

#### 完整pipeline
```
事件序列输入
    ↓
[1] 模型推理 (带TTA)
    ├─ 原始推理
    ├─ 水平翻转推理
    └─ 时间翻转推理 (可选)
    ↓
[2] TTA结果聚合
    └─ NMS + 置信度过滤
    ↓
[3] 基于跟踪的过滤
    ├─ 前向跟踪
    ├─ 后向跟踪 (可选)
    └─ 过滤短轨迹
    ↓
[4] 轨迹填充 (可选)
    └─ 为长轨迹的缺失帧填充bbox
    ↓
[5] 保存伪标签
    └─ 与原始数据集相同格式
```

#### 详细算法

**步骤1: TTA推理** (`predict.py` + `pseudo_labeler.py`)

```python
# predict.py调用
python predict.py model=pseudo_labeler \
    tta.enable=True \
    tta.hflip=True \
    tta.tflip=False  # 可选
```

实现逻辑：
```python
# 1. 准备输入
ev_repr = [B, C, H, W]  # 原始事件表示

# 2. 如果启用hflip
if tta.hflip:
    hflip_ev_repr = torch.flip(ev_repr, dims=[-1])
    ev_repr = torch.cat([ev_repr, hflip_ev_repr], dim=1)  # [2B, C, H, W]
    is_hflip = [False]*B + [True]*B

# 3. 模型推理
predictions = model(ev_repr)  # [2B, N, 7]

# 4. 收集预测
for i, pred in enumerate(predictions):
    if is_hflip[i]:
        pred = flip_back(pred)  # 翻转回原坐标系
    ev_seq_data.update(pred)  # 累积到EventSeqData
```

**步骤2: TTA聚合** (`pseudo_labeler.py:L37-91`)

```python
def tta_postprocess(preds, conf_thre, nms_thre):
    """对多次TTA预测进行聚合"""
    # 1. 置信度过滤
    obj_conf = pred[:, 4]  # objectness
    cls_conf = pred[:, 5]  # class confidence
    mask = (obj_conf * cls_conf >= conf_thre)
    pred = pred[mask]
    
    # 2. Class-wise NMS
    keep = ops.batched_nms(
        boxes=pred[:, :4],           # xyxy
        scores=pred[:, 4] * pred[:, 5],  # obj * cls
        idxs=pred[:, 6],             # class_id
        iou_threshold=nms_thre
    )
    
    return pred[keep]
```

**步骤3: 跟踪过滤** (`pseudo_labeler.py:L202-333`)

核心思想：真实物体应该形成较长的时间轨迹，短轨迹通常是假阳性。

```python
def _track_filter(self):
    """双向跟踪过滤"""
    # 1. 前向跟踪
    remove_idx_fwd, inpaint_bbox = self._track(
        labels=self.labels,
        frame_idx=self.frame_idx,
        min_track_len=6,
        inpaint=True
    )
    
    # 2. 后向跟踪 (可选)
    if 'backward' in track_method:
        rev_labels = [label.reverse() for label in self.labels[::-1]]
        remove_idx_bwd = self._track(rev_labels, ...)
        # 取交集: 两个方向都认为要删除的才删除
        remove_idx = set(remove_idx_fwd) & set(remove_idx_bwd)
    
    # 3. 标记忽略 (不直接删除，而是设class_id=1024)
    for bbox_idx in remove_idx:
        label.class_id[bbox_idx] = 1024  # ignore_label
    
    # 4. 填充缺失帧 (inpainting)
    for f_idx, bbox in inpaint_bbox.items():
        if f_idx in self.frame_idx:
            self.labels[idx] += bbox  # 添加填充的bbox
        else:
            self.frame_idx.append(f_idx)
            self.labels.append(bbox)
```

**步骤4: 保存** (`pseudo_labeler.py:L335-397`)

```python
def save(self, save_dir, dst_name):
    """保存伪标签数据集"""
    # 1. 软链接事件表示 (节省存储)
    os.symlink(original_h5_path, new_h5_path)
    
    # 2. 保存标签 (.npz格式)
    labels = concat_all_labels()  # [N, 8] structured array
    objframe_idx_2_label_idx = compute_indices()
    np.savez(label_path, 
             labels=labels,
             objframe_idx_2_label_idx=objframe_idx_2_label_idx)
    
    # 3. 软链接val/test集
    os.symlink(original_val, new_val)
    os.symlink(original_test, new_test)
```

#### 关键参数调优

| 参数 | 默认值 | 作用 | 调优建议 |
|-----|--------|------|---------|
| `obj_thresh` | 0.01 | objectness阈值 | 越高越严格，减少假阳性 |
| `cls_thresh` | 0.01 | class confidence阈值 | 配合obj_thresh使用 |
| `nms_thre` | 0.45 | NMS的IoU阈值 | 控制重复检测 |
| `min_track_len` | 6 | 最小轨迹长度 | 越大过滤越严格 |
| `inpaint` | True | 是否填充缺失帧 | 可增加标注密度 |
| `track_method` | 'forward' | 跟踪方向 | 'forward or backward'更保守 |

### 3.2 时间上下文建模（LSTM状态管理）

#### 为什么需要Per-Worker状态？

在流式数据加载中，不同worker加载不同的事件序列。如果共享LSTM状态会导致时间上下文混乱。

```
Worker 0: seq_A[0] -> seq_A[1] -> seq_A[2] -> ...
Worker 1: seq_B[0] -> seq_B[1] -> seq_B[2] -> ...
Worker 2: seq_C[0] -> seq_C[1] -> seq_C[2] -> ...
```

每个worker需要维护独立的LSTM状态才能正确建模时间依赖。

#### 实现细节

**状态存储结构** (`modules/utils/detection.py`)
```python
class RNNStates:
    """管理多个worker的RNN状态"""
    def __init__(self):
        self.worker_id_2_states: Dict[int, LstmStates] = {}
        # LstmStates = List[(h, c), (h, c), ...]  # 每个stage一个
    
    def reset(self, worker_id, indices_or_bool_tensor):
        """重置指定worker的状态"""
        if worker_id not in self.worker_id_2_states:
            self.worker_id_2_states[worker_id] = None
            return
        
        # 如果is_first_sample=True，重置对应batch的状态
        states = self.worker_id_2_states[worker_id]
        for stage_idx, (h, c) in enumerate(states):
            h[:, indices_or_bool_tensor] = 0
            c[:, indices_or_bool_tensor] = 0
    
    def save_states_and_detach(self, worker_id, states):
        """保存并detach状态（防止梯度累积）"""
        detached = [(h.detach(), c.detach()) for h, c in states]
        self.worker_id_2_states[worker_id] = detached
```

**训练时的状态流转**
```python
# training_step()
worker_id = batch['worker_id']
is_first_sample = batch['is_first_sample']  # [B], bool

# 1. 重置新序列的状态
self.mode_2_rnn_states[mode].reset(worker_id, is_first_sample)

# 2. 获取上次保存的状态
prev_states = self.mode_2_rnn_states[mode].get_states(worker_id)
# prev_states = [(h0, c0), (h1, c1), (h2, c2), (h3, c3)]

# 3. 逐帧前向传播
for t in range(L):
    ev_repr = ev_tensor_sequence[t]  # [B, C, H, W]
    backbone_features, states = model.forward_backbone(
        x=ev_repr,
        previous_states=prev_states  # 使用上一时刻状态
    )
    prev_states = states  # 更新状态
    # 在有标签的帧收集特征
    if has_label[t]:
        collect_features(backbone_features)

# 4. 保存状态供下次使用
self.mode_2_rnn_states[mode].save_states_and_detach(worker_id, prev_states)
```

#### LSTM状态的维度

```python
# 对于RVT-Small配置
embed_dim = 48
dim_multiplier = [1, 2, 4, 8]
stage_dims = [48, 96, 192, 384]

# 每个stage的LSTM状态
# Stage 1: [B, 48, H/4, W/4]
# Stage 2: [B, 96, H/8, W/8]
# Stage 3: [B, 192, H/16, W/16]
# Stage 4: [B, 384, H/32, W/32]

# 总状态大小 (以Gen1 240x304为例)
# (B, 48, 60, 76) + (B, 96, 30, 38) + (B, 192, 15, 19) + (B, 384, 7, 9)
# ≈ B * (219k + 109k + 54k + 24k) = B * 406k parameters
```

### 3.3 弱监督/半监督学习实现

#### WSOD vs SSOD

**弱监督 (WSOD - Weakly Supervised Object Detection)**:
- 仅使用少量标注数据（如1%, 2%, 5%）
- 配置：`dataset=gen1x0.01` (1%标注)

**半监督 (SSOD - Semi-Supervised Object Detection)**:
- 使用少量标注数据 + 大量无标注数据
- 配置：`dataset=gen1x0.01_ss` (1%标注 + 99%伪标签)

#### 实现策略

**1. 数据配置** (以Gen1为例)

```yaml
# config/dataset/gen1x0.01.yaml - WSOD
dataset:
  name: gen1
  path: ./datasets/gen1
  ratio: 0.01  # 使用1%的标注数据
  train_ratio: 1.0  # 在这1%中使用全部
```

```yaml
# config/dataset/gen1x0.01_ss-1round.yaml - SSOD
dataset:
  name: gen1
  path: ./datasets/pseudo_gen1/gen1x0.01_ss-1round  # 伪标签数据集
  ratio: -1  # 使用所有数据（包括伪标签）
  train_ratio: -1
```

**2. Soft Teacher训练** (`model=rnndet-soft`)

区别于常规训练 (`model=rnndet`)，Soft Teacher使用：
- **Soft Anchor Assignment**: 使用软标签而非硬标签
- **Loss Reweighting**: 根据伪标签质量调整loss权重

```python
# config/model/rnndet-soft.yaml
model:
  head:
    use_soft_label: True  # 启用软标签
    soft_label_cfg:
      temperature: 1.0
      reweight_by_conf: True  # 按置信度重新加权
```

**3. 标签子采样** (`modules/detection.py:L141-147`)

伪标签非常密集（可能每帧都有），但训练时不需要全部使用：

```python
# 每隔k帧使用一次标签
self.label_subsample_idx = get_subsample_label_idx(
    L=sequence_length,  # 序列长度
    use_every=self.mdl_config.get('use_label_every', 1)
)

# 训练时过滤
for tidx in range(len(sparse_obj_labels)):
    if tidx not in self.label_subsample_idx:
        # 保留GT标签，但移除伪标签
        sparse_obj_labels[tidx].set_non_gt_labels_to_none_()
```

#### 训练策略对比

| 阶段 | 模型 | 数据 | Batch Size | Learning Rate | Steps |
|-----|------|------|-----------|---------------|-------|
| Pre-train | rnndet | gen1x0.01 (1% GT) | 8 | 2e-4 | 200k |
| 1st Round | rnndet-soft | gen1x0.01_ss-1round (伪标签) | 8 | 5e-4 | 150k |
| 2nd Round | rnndet-soft | gen1x0.01_ss-2round (伪标签) | 8 | 5e-4 | 150k |

**关键区别**:
1. 伪标签训练使用更高学习率（5e-4 vs 2e-4）
2. 伪标签训练更快收敛（150k vs 200k steps）
3. 原因：伪标签密度更高，有效batch size更大

### 3.4 TTA策略深度分析

#### 为什么TTA有效？

**1. 几何不变性增强**
- **水平翻转**: 物体可能出现在图像左侧或右侧，翻转增强泛化性
- **时间翻转**: 事件相机捕捉运动，反向播放测试时间一致性

**2. 不确定性降低**
- 多次预测的ensemble减少单次预测的噪声
- NMS聚合多个视角的检测结果

#### 实现对比

**推理时TTA** (`val.py` 或 `vis_pred.py`)
```bash
python val.py ... tta.enable=True  # 仅评估时使用
```

仅用于最终评估，不影响训练。

**伪标签生成时TTA** (`predict.py`)
```bash
python predict.py model=pseudo_labeler ... tta.enable=True
```

提高伪标签质量，影响后续训练。**这是关键！**

#### TTA的代价

| TTA配置 | 推理时间 | 显存占用 | 伪标签质量 |
|---------|----------|---------|-----------|
| 无TTA | 1x | 1x | 基线 |
| Hflip only | 2x | 2x | +2-3% AP |
| Hflip + Tflip | 4x | 2x | +3-5% AP |

**推荐**:
- 伪标签生成: 开启 `hflip=True`（代价可接受，收益明显）
- 最终评估: 开启 `hflip=True, tflip=True`（追求最高精度）
- 训练中评估: 关闭TTA（加速训练）

---

## 4. 实验配置与复现指南

### 4.1 配置文件结构

LEOD使用Hydra进行配置管理，采用分层组合：

```
config/
├── general.yaml          # 通用配置 (硬件、日志等)
├── train.yaml            # 训练入口配置
├── model/
│   ├── rnndet.yaml       # 标准检测模型
│   └── rnndet-soft.yaml  # Soft Teacher模型
├── dataset/
│   ├── gen1x0.01.yaml    # Gen1 1%标注
│   ├── gen1x0.01_ss.yaml # Gen1 1%标注 + 伪标签数据集配置
│   └── gen4x0.01_ss.yaml
└── experiment/
    ├── gen1/
    │   ├── small.yaml    # RVT-Small配置
    │   ├── base.yaml     # RVT-Base配置
    │   └── default.yaml  # 具体参数
    └── gen4/
```

**Hydra组合机制**:
```bash
python train.py \
    model=rnndet \                    # -> config/model/rnndet.yaml
    dataset=gen1x0.01_ss \            # -> config/dataset/gen1x0.01_ss.yaml
    +experiment/gen1="small.yaml"     # -> config/experiment/gen1/small.yaml
```

最终配置 = `general.yaml` + `train.yaml` + `model/rnndet.yaml` + `dataset/gen1x0.01_ss.yaml` + `experiment/gen1/small.yaml`

### 4.2 实验配置映射表

#### 论文Table 2: WSOD on Gen1 (1%标注)

| 论文方法 | mAP | 配置命令 |
|---------|-----|---------|
| Supervised | 26.3% | `python train.py model=rnndet dataset=gen1x0.01 +experiment/gen1="small.yaml"` |
| LEOD (1 round) | 32.1% | `python train.py model=rnndet-soft dataset=gen1x0.01_ss-1round +experiment/gen1="small.yaml"` |
| LEOD (2 rounds) | 34.2% | `python train.py model=rnndet-soft dataset=gen1x0.01_ss-2round +experiment/gen1="small.yaml"` |

#### 论文Table 3: SSOD on Gen4 (1%标注)

| 论文方法 | mAP | 配置命令 |
|---------|-----|---------|
| Supervised | 28.5% | `python train.py model=rnndet dataset=gen4x0.01 +experiment/gen4="small.yaml" hardware.gpus=[0,1]` |
| LEOD (1 round) | 35.2% | `python train.py model=rnndet-soft dataset=gen4x0.01_ss-1round +experiment/gen4="small.yaml" hardware.gpus=[0,1]` |

#### 论文Table 4: 不同数据比例

| 数据比例 | Dataset配置 | Training Steps |
|---------|------------|---------------|
| 1% | `gen1x0.01` | 200k |
| 2% | `gen1x0.02` | 300k |
| 5% | `gen1x0.05` | 400k |
| 10% | `gen1x0.10` | 400k |
| 100% | `gen1` | 400k |

**规律**: 数据越少，训练步数越少（防止过拟合）

### 4.3 完整复现流程

#### 步骤0: 环境准备

```bash
# 1. 创建conda环境
conda env create -f environment.yml
conda activate leod

# 2. 下载数据集
# 下载Gen1数据集到 ./datasets/gen1/
# 下载Gen4数据集到 ./datasets/gen4/
# 参考 docs/install.md 获取下载链接

# 3. (可选) 下载预训练权重
mkdir pretrained
# 下载权重到 pretrained/ 目录
```

#### 步骤1: 预训练（监督学习）

**Gen1 (1% labels, 1 GPU)**
```bash
python train.py \
    model=rnndet \
    hardware.gpus=0 \
    dataset=gen1x0.01_ss \
    +experiment/gen1="small.yaml" \
    training.max_steps=200000 \
    training.learning_rate=0.0002 \
    batch_size.train=8 \
    hardware.num_workers.train=8
```

**预期结果**:
- 训练时间: ~24小时 (V100)
- 验证集mAP: ~26-28%
- Checkpoint保存在: `./checkpoint/rnndet_small-gen1x0.01_ss-bs8_iter200k/models/`

**Gen4 (1% labels, 2 GPUs)**
```bash
python train.py \
    model=rnndet \
    hardware.gpus=[0,1] \
    dataset=gen4x0.01_ss \
    +experiment/gen4="small.yaml" \
    training.max_steps=200000 \
    training.learning_rate=0.000346 \
    batch_size.train=12 \
    hardware.num_workers.train=8
```

**预期结果**:
- 训练时间: ~48小时 (2x V100)
- 验证集mAP: ~28-30%

#### 步骤2: 生成伪标签

**Gen1 (第1轮)**
```bash
python predict.py \
    model=pseudo_labeler \
    dataset=gen1x0.01_ss \
    dataset.path=./datasets/gen1/ \
    checkpoint="./checkpoint/rnndet_small-gen1x0.01_ss-bs8_iter200k/models/last.ckpt" \
    hardware.gpus=0 \
    +experiment/gen1="small.yaml" \
    model.postprocess.confidence_threshold=0.01 \
    tta.enable=True \
    tta.hflip=True \
    save_dir=./datasets/pseudo_gen1/gen1x0.01_ss-1round/train
```

**关键参数说明**:
- `model.postprocess.confidence_threshold=0.01`: 低阈值以保留更多候选
- `tta.enable=True`: 启用TTA提高质量
- `save_dir`: 伪标签保存路径

**预期输出**:
```
Pseudo labels saved to: ./datasets/pseudo_gen1/gen1x0.01_ss-1round/
├── train/
│   └── [所有训练序列]
│       ├── event_representations_v2/ (软链接)
│       └── labels_v2/ (新生成的伪标签)
├── val/ (软链接)
└── test/ (软链接)
```

**质量验证**:
```bash
python val_dst.py \
    model=pseudo_labeler \
    dataset=gen1x0.01_ss \
    dataset.path=./datasets/pseudo_gen1/gen1x0.01_ss-1round \
    checkpoint=1 \
    +experiment/gen1="small.yaml"
```

**预期指标**:
- Precision (AP): ~40-50%
- Recall (AR): ~60-70%

#### 步骤3: 第1轮自训练

**Gen1**
```bash
python train.py \
    model=rnndet-soft \
    hardware.gpus=0 \
    dataset=gen1x0.01_ss-1round \
    +experiment/gen1="small.yaml" \
    training.max_steps=150000 \
    training.learning_rate=0.0005 \
    batch_size.train=8
```

**关键变化**:
- `model=rnndet-soft`: 使用Soft Teacher
- `training.learning_rate=0.0005`: 更高学习率 (原来0.0002)
- `training.max_steps=150000`: 更少步数 (原来200k)

**预期结果**:
- 训练时间: ~18小时
- 验证集mAP: ~32-34%

#### 步骤4: (可选) 第2轮自训练

重复步骤2和3：
1. 用第1轮训练的模型生成新的伪标签
2. 保存到 `./datasets/pseudo_gen1/gen1x0.01_ss-2round/`
3. 在新伪标签上训练

**预期收益**:
- 第2轮: +1-2% mAP
- 第3轮: +0-1% mAP (收益递减)

### 4.4 评估与可视化

#### 最终评估

```bash
# Gen1测试集
python val.py \
    model=rnndet \
    dataset=gen1 \
    dataset.path=./datasets/gen1/ \
    checkpoint="path/to/best_model.ckpt" \
    use_test_set=1 \
    hardware.gpus=0 \
    +experiment/gen1="small.yaml" \
    model.postprocess.confidence_threshold=0.001 \
    reverse=False \
    tta.enable=True \
    tta.hflip=True
```

**注意**: `confidence_threshold=0.001` 用于最终评估（论文中提到影响1-3% mAP）

#### 可视化预测

```bash
python vis_pred.py \
    model=rnndet \
    dataset=gen1 \
    dataset.path=./datasets/gen1/ \
    checkpoint="path/to/model.ckpt" \
    +experiment/gen1="small.yaml" \
    model.postprocess.confidence_threshold=0.1 \
    num_video=5 \
    reverse=False
```

**输出**: MP4视频保存在 `./vis/gen1_rnndet_small/pred/`

### 4.5 故障排查

#### 常见错误1: Shape mismatch when loading checkpoint

**原因**: 模型大小不匹配
```
AssertionError: backbone.stages.0.downsample_cf2cl.conv.weight - 
expected [48, 20, 4, 4], got [96, 20, 4, 4]
```

**解决**: 检查 `+experiment/gen1="?.yaml"`
- RVT-Small: `small.yaml` (embed_dim=48)
- RVT-Base: `base.yaml` (embed_dim=96)

#### 常见错误2: 伪标签格式错误

**原因**: dataset配置与实际路径不匹配

**解决**:
```bash
# 检查数据集路径
ls ./datasets/pseudo_gen1/gen1x0.01_ss-1round/train/

# 确保包含:
# - event_representations_v2/ (软链接)
# - labels_v2/ (新标签)
```

#### 常见错误3: OOM (Out of Memory)

**原因**: Batch size或sequence length过大

**解决**:
```bash
# 减少batch size
batch_size.train=4  # 原来8

# 或减少sequence length
dataset.sequence_length=10  # 原来20
```

---

## 5. 代码质量与创新亮点

### 5.1 代码架构优势

#### 1. 模块化设计

**分离关注点**:
```
modules/
├── detection.py          # 核心检测逻辑
├── pseudo_labeler.py     # 伪标签生成 (继承detection)
└── tracking/             # 跟踪模块

models/
├── detection/
│   ├── recurrent_backbone/   # 可插拔的backbone
│   ├── yolox/                 # 检测头
│   └── yolox_extension/       # LEOD扩展
└── layers/                    # 通用层 (RNN, MaxViT)
```

**优势**:
- `PseudoLabeler`继承`Module`，复用所有检测逻辑
- 只需重写`predict_step`添加TTA和过滤
- 易于扩展到其他检测器或backbone

#### 2. Hydra配置系统

**优势**:
- 配置版本化：每个实验的配置完整保存
- 组合式配置：model + dataset + experiment 自由组合
- 命令行覆盖：`python train.py dataset.ratio=0.02` 无需修改文件

**示例**:
```yaml
# 基础配置
defaults:
  - model: rnndet
  - dataset: gen1x0.01_ss
  - _self_

# 命令行覆盖
# python train.py dataset.ratio=0.05
```

#### 3. PyTorch Lightning集成

**优势**:
- 自动DDP: 多GPU训练无需手动处理
- Checkpoint管理: 自动保存和恢复
- 日志集成: WandB无缝集成
- 回调系统: 可视化、梯度监控等

**关键代码**:
```python
# modules/detection.py
class Module(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        # 自动处理: backward, optimizer.step, logging
        return {'loss': loss}
    
    def configure_optimizers(self):
        # 自动管理optimizer和scheduler
        return optimizer, scheduler
```

### 5.2 工程优化技巧

#### 1. SLURM集群适配

**问题**: 抢占式集群会中断任务

**解决**: 自动检测和恢复checkpoint
```python
# train.py:L71-95
def detect_ckpt(ckpt_path):
    """自动检测最新checkpoint"""
    ckp_files = glob_all(ckpt_path)
    ckp_files = sort_file_by_time(ckp_files)
    last_ckpt = ckp_files[-1]
    # 验证checkpoint完整性
    try:
        torch.load(last_ckpt)
    except:
        last_ckpt = ckp_files[-2]  # 用倒数第二个
    return last_ckpt
```

**亮点**: 
- 自动检测上一个任务的SLURM_JOB_ID
- 恢复WandB run（保持日志连续性）
- 软链接到`/checkpoint`临时目录（避免配额限制）

#### 2. 混合数据加载优化

**挑战**: 如何高效组合random和stream采样？

**解决**: 独立的dataloader + 动态batch合并
```python
# modules/data/genx.py:L120-144
def set_mixed_sampling_mode_variables_for_train(self):
    # 按权重分配batch size
    bs_rnd = round(total_bs * w_random / (w_stream + w_random))
    bs_str = total_bs - bs_rnd
    
    # random采样更慢，分配更多workers
    workers_rnd = ceil(total_workers * bs_rnd / total_bs)
    workers_str = total_workers - workers_rnd
```

**亮点**: 
- 自动平衡两种采样模式的资源
- Random采样得到更多workers（更慢）
- 训练时透明合并batch

#### 3. 内存管理

**问题**: LSTM状态累积导致内存泄漏

**解决**: Detach历史状态
```python
# modules/detection.py:L227-228
self.mode_2_rnn_states[mode].save_states_and_detach(
    worker_id=worker_id, states=prev_states
)

# 实现
def save_states_and_detach(self, worker_id, states):
    # 关键: detach()切断梯度链
    detached = [(h.detach(), c.detach()) for h, c in states]
    self.worker_id_2_states[worker_id] = detached
```

**亮点**: 只保留当前时刻梯度，历史状态不参与反向传播

#### 4. 标签子采样

**问题**: 伪标签太密集，训练慢且无收益

**解决**: 动态子采样
```python
# modules/detection.py:L141-147
for tidx in range(len(sparse_obj_labels)):
    if tidx not in self.label_subsample_idx:
        # 保留GT，移除伪标签
        sparse_obj_labels[tidx].set_non_gt_labels_to_none_()
```

**亮点**: 
- GT标签始终保留
- 伪标签按`use_label_every`间隔采样
- 减少计算开销，不影响性能

### 5.3 创新实现亮点

#### 1. 轨迹填充 (Inpainting)

**论文未明确提及，但代码实现**:
```python
# pseudo_labeler.py:L240-265
for tracker in model.prev_trackers:
    if tracker.hits >= min_track_len:
        for f_idx, bbox in tracker.missed_bbox.items():
            # 在跟踪器丢失的帧填充bbox
            inpainted_bbox[f_idx].append(bbox)
```

**优势**:
- 增加标注密度
- 提高时间一致性
- 特别对快速运动物体有效

#### 2. 双向跟踪

**代码扩展了论文方法**:
```python
# pseudo_labeler.py:L285-298
# 前向跟踪
remove_idx_fwd = self._track(labels, frame_idx)
# 后向跟踪
remove_idx_bwd = self._track(labels[::-1], frame_idx[::-1])
# 取交集: 更保守的过滤
remove_idx = set(remove_idx_fwd) & set(remove_idx_bwd)
```

**优势**:
- 双向一致性检验
- 减少假阳性
- 提高伪标签精度

#### 3. Ignore Label机制

**不删除低质量bbox，而是标记忽略**:
```python
# 设置class_id=1024，训练时跳过
bbox.class_id = 1024  # ignore_label

# 训练时过滤
if class_id == 1024:
    continue  # 不计算loss
```

**优势**:
- 保持数据结构完整
- 便于调试和分析
- 易于调整ignore策略

#### 4. 伪标签质量验证

**自动验证生成的伪标签**:
```bash
python val_dst.py ...  # 计算伪标签的AP和AR
```

```python
# val_dst.py 核心逻辑
# 1. 加载伪标签作为"预测"
pseudo_labels = load_pseudo_labels()
# 2. 加载GT作为"标签"（仅标注帧）
gt_labels = load_gt_labels()
# 3. 计算AP (precision) 和 AR (recall)
metrics = evaluate(pseudo_labels, gt_labels)
```

**亮点**:
- 提前发现低质量伪标签
- 论文建议: AP < 40% 则不进行下一轮训练
- 节省计算资源

### 5.4 与论文对比的改进

| 方面 | 论文描述 | 代码实现 | 改进 |
|------|---------|---------|------|
| 跟踪方向 | 单向跟踪 | 支持双向跟踪 | 更鲁棒 |
| 轨迹处理 | 过滤短轨迹 | 过滤+填充 | 更密集的标注 |
| Ignore Label | 未提及 | class_id=1024机制 | 更灵活 |
| 质量评估 | 论文分析 | `val_dst.py`脚本 | 自动化 |
| 可视化 | 论文展示 | `vis_pred.py` | 便于调试 |
| 配置管理 | - | Hydra系统 | 可复现性强 |

### 5.5 潜在改进方向

#### 1. 自适应阈值

**当前**: 固定`obj_thresh`, `cls_thresh`

**改进**: 根据训练轮次动态调整
```python
# 伪代码
if round == 1:
    obj_thresh = 0.3  # 严格过滤
elif round == 2:
    obj_thresh = 0.2  # 逐渐放宽
```

**收益**: 第1轮高精度，第2轮高召回

#### 2. 类别平衡

**当前**: 所有类别统一阈值

**改进**: 难检测类别（如行人）使用更低阈值
```python
# 伪代码
thresh_per_class = {
    'pedestrian': 0.1,  # 难检测
    'car': 0.3          # 易检测
}
```

**收益**: 提高小目标和遮挡目标的召回率

#### 3. 不确定性估计

**当前**: 仅使用confidence score

**改进**: 引入MC-Dropout或ensemble估计不确定性
```python
# 伪代码
predictions = []
for _ in range(5):  # MC-Dropout采样
    pred = model(x, training=True)
    predictions.append(pred)
uncertainty = std(predictions)  # 计算方差
```

**收益**: 更准确地过滤低质量预测

#### 4. 课程学习

**当前**: 所有伪标签同等对待

**改进**: 先训练高置信度样本，逐步加入低置信度
```python
# 伪代码
if epoch < 50:
    use_samples_with_conf > 0.7
elif epoch < 100:
    use_samples_with_conf > 0.5
else:
    use_all_samples
```

**收益**: 更稳定的训练，减少噪声影响

### 5.6 代码可扩展性

#### 易于扩展到其他数据集

**当前**: 支持Gen1, Gen4

**扩展**: 添加新数据集只需3步
```python
# 1. 添加dataset config
# config/dataset/my_dataset.yaml
dataset:
  name: my_dataset
  path: ./datasets/my_dataset
  ...

# 2. (可选) 添加bbox过滤逻辑
# modules/utils/ssod.py
def filter_pred_boxes(bbox, dataset_name):
    if dataset_name == 'my_dataset':
        # 数据集特定的过滤
        return filtered_bbox

# 3. 添加experiment config
# config/experiment/my_dataset/small.yaml
```

#### 易于扩展到其他检测器

**当前**: YOLOX

**扩展**: 替换检测头
```python
# models/detection/yolox_extension/models/detector.py
class YoloXDetector(nn.Module):
    def __init__(self, mdl_config, ssod=False):
        self.backbone = RNNDetector(...)  # 保持不变
        self.neck = PAFPN(...)            # 保持不变
        self.head = YOLOXHead(...)        # 替换为其他检测头
```

**可选检测头**: FCOS, ATSS, Faster R-CNN等

---

## 总结

### 核心贡献

1. **标签高效学习范式**: 将self-training成功应用于事件相机目标检测
2. **高质量伪标签生成**: TTA + 跟踪过滤 + 填充的完整pipeline
3. **时间建模**: Recurrent backbone + per-worker状态管理
4. **工程实现**: 模块化、可复现、易扩展的代码库

### 实用价值

- **学术**: 为事件相机的弱监督学习提供baseline
- **工业**: 减少标注成本（1%标注达到~40%的性能）
- **教育**: 优秀的PyTorch Lightning项目模板

### 建议使用场景

**适合**:
- 标注预算有限（< 10%数据）
- 事件相机应用（自动驾驶、机器人等）
- 需要时间建模的序列数据

**不适合**:
- 已有充足标注（self-training收益有限）
- 静态图像检测（无需recurrent建模）
- 实时性要求极高（LSTM有延迟）

---

## 参考资源

- **论文**: [LEOD: Label-Efficient Object Detection for Event Cameras](https://arxiv.org/abs/2311.17286)
- **代码**: [GitHub - LEOD](https://github.com/Wuziyi616/LEOD)
- **数据集**: [Prophesee Gen1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/), [Gen4](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/)
- **相关工作**: [RVT](https://github.com/uzh-rpg/RVT), [Soft Teacher](https://github.com/microsoft/SoftTeacher)

---

*本文档由CTO.new AI助手生成，用于帮助中文用户理解LEOD论文与代码的对应关系。*
