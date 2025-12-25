# LEOD: 事件相机标签高效目标检测 - 架构文档

## 第一部分：项目概述

### LEOD项目简介

LEOD（Label-Efficient Object Detection for Event Cameras）是一个针对事件相机的标签高效目标检测框架。事件相机是一种生物启发的低延迟传感器，具有极高的时间分辨率（>1000 FPS），在自动驾驶等安全关键应用中具有巨大潜力。

### 关键特点和创新点

1. **混合采样策略**：结合随机访问和流式数据加载，平衡训练效率和时序连续性
2. **RNN状态管理**：跨批次保持LSTM状态，确保事件序列的时序连续性
3. **伪标签自训练**：利用模型预测生成伪标签，实现弱监督/半监督学习
4. **MaxViT-RNN骨干网络**：结合MaxViT和RNN的混合架构，有效处理事件数据
5. **PAFPN特征融合**：改进的特征金字塔网络，增强多尺度特征表示
6. **YOLOX检测头**：高效的目标检测头，适用于事件相机数据

### 论文和项目链接

- **论文**：[LEOD: Label-Efficient Object Detection for Event Cameras](https://arxiv.org/abs/2311.17286)
- **GitHub**：[https://github.com/Wuziyi616/LEOD](https://github.com/Wuziyi616/LEOD)
- **arXiv**：[https://arxiv.org/abs/2311.17286](https://arxiv.org/abs/2311.17286)

## 第二部分：核心架构总览

### 完整的目录结构树形图

```
LEOD/
├── config/                  # Hydra配置文件
│   ├── general.yaml         # 通用训练配置
│   ├── dataset/             # 数据集配置
│   ├── experiment/          # 实验预设
│   ├── model/               # 模型配置
│   └── modifier.py          # 配置动态修改
├── modules/                 # 核心模块
│   ├── detection.py         # 检测模块（主模型）
│   ├── pseudo_labeler.py    # 伪标签生成器
│   ├── data/                # 数据模块
│   │   └── genx.py          # 数据加载和处理
│   └── utils/               # 工具函数
├── models/                  # 模型定义
│   └── detection/           # 检测相关模型
│       ├── recurrent_backbone/  # RNN骨干网络
│       └── yolox_extension/     # YOLOX扩展
│           └── models/          # 检测器定义
│               └── detector.py  # YoloXDetector
├── data/                    # 数据处理
│   └── genx_utils/          # 数据工具
│       ├── labels.py        # 标签处理
│       ├── dataset_rnd.py   # 随机访问数据集
│       └── dataset_streaming.py  # 流式数据集
├── train.py                 # 训练入口
├── predict.py               # 推理和伪标签生成
├── val.py                   # 验证
└── val_dst.py               # 伪标签质量评估
```

### 各目录的功能说明

| 目录 | 功能描述 |
|------|----------|
| `config/` | Hydra配置系统，管理所有实验参数和超参数 |
| `modules/` | 核心模块，包括检测模型、伪标签生成器和数据处理 |
| `models/` | 神经网络模型定义，特别是检测相关模型 |
| `data/` | 数据处理和加载逻辑 |
| `train.py` | 主训练脚本，处理DDP、检查点、日志等 |
| `predict.py` | 推理脚本，支持TTA和伪标签生成 |
| `val.py` | 验证脚本，计算Prophesee AP指标 |
| `val_dst.py` | 伪标签质量评估脚本 |

### 模块间的关系图

```
┌───────────────────────────────────────────────────────┐
│                   训练流程 (train.py)                  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              Hydra配置加载 (config/general.yaml)       │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              数据模块 (modules/data/genx.py)            │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 随机访问数据集   │    │ 流式数据集             │  │
│  │ (dataset_rnd.py)│    │ (dataset_streaming.py)│  │
│  └─────────────────┘    └─────────────────────────┘  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              检测模块 (modules/detection.py)            │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ YoloXDetector   │    │ RNN状态管理            │  │
│  │ (detector.py)   │    │ (RNNStates)            │  │
│  └─────────────────┘    └─────────────────────────┘  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              模型架构 (models/detection/)              │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ MaxViT-RNN骨干  │    │ PAFPN特征融合          │  │
│  │ (recurrent_backbone)│  │ (yolo_pafpn.py)        │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐                                │
│  │ YOLOX检测头     │                                │
│  │ (yolox_head.py) │                                │
│  └─────────────────┘                                │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              伪标签生成 (modules/pseudo_labeler.py)     │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ TTA增强         │    │ 追踪后处理             │  │
│  │ (Test-Time Aug) │    │ (LinearTracker)        │  │
│  └─────────────────┘    └─────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

## 第三部分：关键模块详解

### 1. 检测模块 (modules/detection.py)

#### 类结构说明

```python
class Module(pl.LightningModule):
    """基于事件的检测模块，包含：
    - 循环骨干网络（MaxViT-RNN）
    - 检测头（YOLOX）
    """
```

#### 主要方法的功能

| 方法 | 功能描述 |
|------|----------|
| `__init__` | 初始化模型，加载配置，创建YoloXDetector实例 |
| `setup` | 设置训练/验证/测试阶段的参数和评估器 |
| `forward` | 前向传播，处理事件张量和RNN状态 |
| `training_step` | 单步训练逻辑，处理混合采样批次 |
| `validation_step` | 验证逻辑，流式评估 |
| `test_step` | 测试逻辑，流式评估 |
| `configure_optimizers` | 配置优化器和学习率调度器 |

#### RNN状态管理机制

```python
self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
    Mode.TRAIN: RNNStates(),
    Mode.VAL: RNNStates(),
    Mode.TEST: RNNStates(),
}
```

RNNStates容器负责：
1. **状态重置**：根据`is_first_sample`标志重置特定worker的RNN状态
2. **状态保存**：在每个批次结束时保存和分离RNN状态
3. **状态检索**：为每个worker检索之前的RNN状态
4. **跨批次连续性**：确保事件序列的时序连续性

#### 混合采样处理

```python
def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
    batch = merge_mixed_batches(batch)  # 合并随机和流式批次
    data = self.get_data_from_batch(batch)
    worker_id = self.get_worker_id_from_batch(batch)
    
    # 处理混合采样逻辑
    if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
        predictions = predictions[-B:]  # 仅评估最后批次
        obj_labels = obj_labels[-B:]
```

#### 实际代码片段

**前向传播和RNN状态管理**：

```python
def forward(self,
            event_tensor: th.Tensor,
            previous_states: Optional[LstmStates] = None,
            retrieve_detections: bool = True,
            targets=None) -> Tuple[th.Tensor, Dict[str, th.Tensor], LstmStates]:
    return self.mdl(x=event_tensor,
                    previous_states=previous_states,
                    retrieve_detections=retrieve_detections,
                    targets=targets)

# RNN状态重置
self.mode_2_rnn_states[mode].reset(
    worker_id=worker_id, 
    indices_or_bool_tensor=is_first_sample)

# RNN状态保存
self.mode_2_rnn_states[mode].save_states_and_detach(
    worker_id=worker_id, states=prev_states)
```

**训练步骤核心逻辑**：

```python
# 主干网络前向传播
backbone_features, states = self.mdl.forward_backbone(
    x=ev_tensors,
    previous_states=prev_states,
    token_mask=token_masks)

# 检测头前向传播
predictions, losses = self.mdl.forward_detect(
    backbone_features=selected_backbone_features, 
    targets=labels_yolox)

# 损失计算和日志记录
output = {'loss': losses['loss']}
self.log_dict({f'{prefix}{k}': v for k, v in losses.items()}, 
              on_step=True, on_epoch=True, batch_size=B)
```

### 2. 伪标签生成器 (modules/pseudo_labeler.py)

#### 自训练框架工作流程

```
┌───────────────────────────────────────────────────────┐
│              伪标签生成流程                            │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              模型推理 (forward)                        │
│  - 完整序列推理                                      │
│  - TTA增强 (可选)                                    │
│  - 置信度过滤                                        │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              后处理 (postprocess)                     │
│  - NMS (非极大值抑制)                                │
│  - 追踪过滤 (LinearTracker)                          │
│  - 插值补全                                         │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              质量评估 (evaluate_label)                 │
│  - 精度计算                                         │
│  - 召回率计算                                       │
│  - AP指标                                           │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              保存伪标签 (EventSeqData.save)            │
│  - 软链接事件数据                                   │
│  - 保存标签为.npz文件                               │
│  - 保存索引映射                                     │
└───────────────────────────────────────────────────────┘
```

#### TTA集成方式

```python
# TTA配置
self.tta_cfg = self.full_config.tta
if self.tta_cfg.enable:
    print('Using TTA in pseudo label generation.')

# 水平翻转增强
def get_data_from_batch(self, batch: Any):
    if self.tta_cfg.enable and self.tta_cfg.hflip:
        hflip_ev_repr = th.flip(ev_repr, dims=[-1])
        ev_repr = th.cat([ev_repr, hflip_ev_repr], dim=1)  # 2B
        # 复制其他数据项
        for k in (DataType.IS_FIRST_SAMPLE, DataType.IS_LAST_SAMPLE, DataType.IS_REVERSED):
            new_data[k] = th.cat([data[k]] * 2, dim=-1)
        # 标签水平翻转
        for i, (lbl, lbl_flip) in enumerate(zip(labels, labels_flip)):
            lbl_flip.flip_lr_()
            labels[i] = lbl + lbl_flip
```

#### 追踪后处理逻辑

```python
class LinearTracker:
    """线性追踪器，用于过滤短轨迹和插值补全缺失检测。"""
    
    def update(self, frame_idx, dets, is_gt):
        """更新追踪器状态"""
        # 分配检测到现有轨迹或创建新轨迹
        
    def finish(self):
        """完成追踪，处理未完成轨迹"""
        
    def get_bbox_tracker(self, bbox_idx):
        """获取特定边界框的追踪器"""

# 追踪过滤逻辑
def _track_filter(self):
    """双向追踪过滤"""
    # 前向追踪
    remove_idx, inpainted_bbox = self._track(
        self.labels, self.frame_idx, 
        min_track_len=min_track_len, inpaint=self.filter_config.inpaint)
    
    # 后向追踪
    if 'backward' in track_method:
        rev_labels = [label.get_reverse() for label in self.labels[::-1]]
        rev_frame_idx = [max(self.frame_idx) - idx for idx in self.frame_idx[::-1]]
        bg_remove_idx, _ = self._track(rev_labels, rev_frame_idx, min_track_len=min_track_len)
        
    # 移除短轨迹检测
    for idx, obj_label in enumerate(self.labels):
        for i in range(len(obj_label)):
            if bbox_idx in remove_idx:
                new_class_id[i] = self.filter_config.ignore_label
```

#### 质量评估方法

```python
def evaluate_label(pred_labels, gt_labels, obj_thresh, cls_thresh):
    """评估伪标签质量"""
    # 计算精度、召回率、AP等指标
    
    # 过滤低置信度预测
    pred_labels = filter_pred_boxes(pred_labels, obj_thresh, cls_thresh)
    
    # 计算IoU和匹配
    ious = box_iou(pred_labels.bboxes, gt_labels.bboxes)
    
    # 计算精度和召回率
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return {'precision': precision, 'recall': recall, 'ap': ap}
```

#### 伪标签生成流程图

```
┌───────────────────────────────────────────────────────┐
│              EventSeqData.update()                     │
│  - 收集模型预测                                      │
│  - 应用数据增强 (TTA)                                │
│  - 存储帧索引到标签的映射                            │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              EventSeqData._aggregate_results()         │
│  - 合并TTA结果                                        │
│  - 按帧索引排序                                      │
│  - 应用NMS                                           │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              EventSeqData._track_filter()              │
│  - 双向追踪 (前向+后向)                              │
│  - 过滤短轨迹 (<6帧)                                 │
│  - 插值补全缺失检测                                  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              EventSeqData._summarize()                 │
│  - 转换为结构化数组                                  │
│  - 创建索引映射                                      │
│  - 准备保存格式                                      │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              EventSeqData.save()                       │
│  - 软链接事件数据                                    │
│  - 保存标签为labels.npz                              │
│  - 保存索引映射                                      │
│  - 链接验证/测试集                                   │
└───────────────────────────────────────────────────────┘
```

### 3. 数据模块 (modules/data/genx.py)

#### DataModule类介绍

```python
class DataModule(pl.LightningDataModule):
    """事件检测数据集/数据加载器的基础数据模块。
    
    有两种可能的数据集：随机访问和流式。
    - 随机访问：标签帧和事件表示从整个事件序列中随机采样
    - 流式：数据以顺序方式加载，保持时序连续性
    """
```

#### 随机访问 vs 流式加载对比

| 特性 | 随机访问 | 流式加载 |
|------|----------|----------|
| **采样方式** | 随机采样 | 顺序加载 |
| **时序连续性** | 无 | 有 |
| **批处理** | 标准批处理 | 无批处理（单序列） |
| **数据增强** | 支持 | 不支持 |
| **加载速度** | 快 | 慢 |
| **内存使用** | 低 | 高 |
| **适用场景** | 预训练 | 微调、评估 |

#### 混合采样模式详解

```python
def set_mixed_sampling_mode_variables_for_train(self):
    """确定有多少样本是随机数据/流式数据。"""
    # 根据权重设置批大小
    bs_rnd = min(round(self.overall_batch_size_train * weight_random / (weight_stream + weight_random)),
                  self.overall_batch_size_train - 1)
    bs_str = self.overall_batch_size_train - bs_rnd
    
    # 根据批大小设置worker数量
    workers_rnd = min(math.ceil(self.overall_num_workers_train * bs_rnd / self.overall_batch_size_train),
                       self.overall_num_workers_train - 1)
    workers_str = self.overall_num_workers_train - workers_rnd
```

混合采样策略：
1. **动态权重分配**：根据配置的权重（`w_random`, `w_stream`）动态分配批大小
2. **资源优化**：随机采样通常需要更多worker，因为数据加载更慢
3. **批次合并**：在训练步骤中合并来自不同采样模式的批次
4. **RNN状态管理**：为每种采样模式单独管理RNN状态

#### 数据批次结构

```python
# 随机访问批次结构
{
    'event_repr': List[Tensor],  # [L, B, C, H, W] 事件表示
    'obj_labels': List[ObjectLabels],  # [L, B] 标签
    'is_first_sample': Tensor,  # [B] 是否为序列首帧
    'path': List[str],  # [B] 数据路径
    'ev_idx': List[Tensor],  # [L, B] 事件索引
}

# 流式批次结构
{
    'event_repr': List[Tensor],  # [L, 1, C, H, W] 单序列事件
    'obj_labels': List[ObjectLabels],  # [L, 1] 单序列标签
    'is_first_sample': Tensor,  # [1] 是否为序列首帧
    'is_last_sample': Tensor,  # [1] 是否为序列末帧
    'path': List[str],  # [1] 数据路径
}
```

#### 多worker并行加载策略

```python
def get_dataloader_kwargs(dataset, sampling_mode, dataset_mode, dataset_config, batch_size, num_workers):
    """为不同采样模式获取数据加载器参数。"""
    
    if sampling_mode == DatasetSamplingMode.STREAM:
        return dict(
            dataset=dataset,
            batch_size=None,  # 无批处理！
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_streaming,
        )
    
    elif sampling_mode == DatasetSamplingMode.RANDOM:
        sampler = get_weighted_random_sampler(dataset) if use_weighted_rnd_sampling else None
        return dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_rnd,
        )
```

### 4. 标签管理 (data/genx_utils/labels.py)

#### ObjectLabels类说明

```python
class ObjectLabels(ObjectLabelBase):
    """表示N个边界框标签，形状为[N, num_fields (8)]。
    
    类似于torch.Tensor，具有dtype、device等属性，以及边界框属性（x,y,w,h,class_id）。
    **边界框格式为角落格式！即x,y是左上角坐标。**
    """
```

#### 数据格式和转换

```python
BBOX_DTYPE = np.dtype({
    'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'class_confidence', 'objectness'],
    'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<f4', '<f4'],
    'offsets': [0, 8, 12, 16, 20, 24, 28, 32],
    'itemsize': 40
})

# 转换方法
to_structured_array()  # 转换为结构化数组
get_labels_as_tensors(format_='prophesee')  # 转换为张量
get_labels_as_batched_tensor(format_='yolox')  # 转换为YOLOX格式批次张量
```

#### SparselyBatchedObjectLabels使用

```python
class SparselyBatchedObjectLabels:
    """表示一个批次中稀疏标记的对象标签。
    
    处理每个批次中不同帧的标签可用性。
    """
    
    def get_valid_labels_and_batch_indices(self, ignore=False, ignore_label=1024):
        """获取有效标签和对应的批次索引"""
        # 返回：current_labels (List[ObjectLabels]), valid_batch_indices (List[int])
    
    def set_non_gt_labels_to_none_(self):
        """将非GT标签设置为None，用于训练加速"""
```

### 5. 检测头架构 (models/detection/yolox_extension/models/detector.py)

#### YoloXDetector类结构

```python
class YoloXDetector(th.nn.Module):
    """基于RNN的MaxViT骨干 + YOLOX检测头。"""
    
    def __init__(self, model_cfg: DictConfig, ssod: bool = False):
        super().__init__()
        
        # 构建骨干网络
        self.backbone = build_recurrent_backbone(backbone_cfg)  # maxvit_rnn
        
        # 构建特征金字塔网络
        in_channels = self.backbone.get_stage_dims(fpn_cfg.in_stages)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)
        
        # 构建检测头
        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        self.yolox_head = build_yolox_head(head_cfg, in_channels=in_channels, strides=strides, ssod=ssod)
```

#### MaxViT-RNN骨干网络

```python
# 架构特点
1. **多阶段特征提取**：提取多尺度特征（stage 1-4）
2. **RNN状态管理**：每个阶段维护独立的LSTM状态
3. **时序连续性**：跨批次保持RNN状态
4. **高效计算**：支持torch.compile优化

# 前向传播
backbone_features, states = self.backbone(x, previous_states, token_mask)
# backbone_features: Dict{stage_id: feats, [B, C, h, w]}
# states: List[(lstm_h, lstm_c), same shape]
```

#### PAFPN特征融合

```python
# PAFPN (Path Aggregation FPN) 特点
1. **自顶向下路径**：高层特征到低层特征的融合
2. **自底向上路径**：低层特征到高层特征的融合
3. **跨尺度连接**：增强多尺度特征表示
4. **高效计算**：优化的特征金字塔网络

# 前向传播
fpn_features = self.fpn(backbone_features)  # Tuple(feats, [B, C, h, w])
```

#### YOLOX检测头

```python
# YOLOX检测头特点
1. **无锚点设计**：直接预测边界框坐标
2. **解耦头**：分离分类和定位任务
3. **高效NMS**：优化的非极大值抑制
4. **多尺度预测**：在不同特征层级进行预测

# 前向传播
outputs, losses = self.yolox_head(fpn_features, targets, soft_targets)
# outputs: (B, N, 4 + 1 + num_cls), [(x, y, w, h), obj_conf, cls]
# losses: Dict{loss_name: loss, torch.scalar tensor}
```

#### 完整的前向传播路径

```python
def forward(self,
            x: th.Tensor,
            previous_states: Optional[LstmStates] = None,
            retrieve_detections: bool = True,
            targets: Optional[th.Tensor] = None) -> 
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
    
    # 1. 主干网络前向传播
    backbone_features, states = self.forward_backbone(x, previous_states)
    
    # 2. 特征金字塔网络
    fpn_features = self.fpn(backbone_features)
    
    # 3. 检测头
    if retrieve_detections:
        outputs, losses = self.yolox_head(fpn_features, targets)
        return outputs, losses, states
    
    return None, None, states
```

## 第四部分：训练流程详解

### 1. 初始化流程 (train.py)

#### Hydra配置加载

```python
@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    # 动态修改配置
    dynamically_modify_train_config(config)
    
    # 解析配置
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    
    # 打印配置
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')
```

#### DDP分布式设置

```python
# 配置DDP策略
gpu_config = config.hardware.gpus
gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
gpus = gpus if isinstance(gpus, list) else [gpus]

distributed_backend = config.hardware.dist_backend
assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'

strategy = DDPStrategy(
    process_group_backend=distributed_backend,
    find_unused_parameters=False,
    gradient_as_bucket_view=True) if len(gpus) > 1 else None
```

#### 检查点管理（Slurm优化）

```python
def detect_ckpt(ckpt_path: str):
    """在SLURM抢占系统中自动检测检查点。"""
    last_ckpt = None
    
    if os.path.exists(ckpt_path):
        ckp_files = glob_all(ckpt_path)
        ckp_files = [ckp for ckp in ckp_files if ckp.endswith('.ckpt')]
        if ckp_files:
            ckp_files = sort_file_by_time(ckp_files)  # 0-th is oldest
            last_ckpt = ckp_files[-1]
            try:
                _ = torch.load(last_ckpt, map_location='cpu')
            except:
                os.remove(last_ckpt)
                last_ckpt = None
                if len(ckp_files) > 1:
                    last_ckpt = ckp_files[-2]
            print(f'INFO: automatically detect checkpoint {last_ckpt}')
    
    return last_ckpt

# 集群特定优化
SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
if SLURM_JOB_ID and os.path.isdir('/checkpoint/'):
    # 软链接临时空间用于检查点
    usr = pwd.getpwuid(os.getuid())[0]
    new_dir = f'/checkpoint/{usr}/{SLURM_JOB_ID}/'
    
    # 检查点目录可能已存在，这意味着我们正在恢复训练
    if os.path.exists(ckpt_dir):
        old_slurm_id = find_old_slurm_id(ckpt_dir)
        if old_slurm_id is None:
            slurm_id = SLURM_JOB_ID
        wandb_name = f'{exp_name}-{slurm_id}'
        
        # 将所有内容移动到新目录，因为旧目录可能被清除
        if str(old_slurm_id) != str(SLURM_JOB_ID):
            for f in sort_file_by_time(glob_all(ckpt_dir)):
                if 'SLURM_JOB_FINISHED' in f:
                    os.system(f'rm -f {f}')
                else:
                    os.system(f'mv {f} {new_dir}')
        os.system(f'rm -rf {ckpt_dir}')
    
    os.system(f'ln -s {new_dir} {ckpt_dir}')
    os.system(f"touch {os.path.join(ckpt_dir, 'DELAYPURGE')}")
```

#### WandB日志配置

```python
# 配置WandB日志
config.wandb.wandb_name = wandb_name
config.wandb.wandb_id = wandb_id
config.wandb.wandb_runpath = ckpt_dir
config.wandb.group_name = config.dataset.name

logger = get_wandb_logger(config)

# 自动检测检查点
ckpt_path = detect_ckpt(ckpt_dir)
if not ckpt_path and config.checkpoint:
    ckpt_path = config.checkpoint
    print(f'INFO: use pre-specified checkpoint {ckpt_path}')

# 模型监控
logger.watch(model=module, log='all', 
             log_freq=config.logging.train.log_model_every_n_steps, 
             log_graph=True)
```

#### 详细代码示例

**完整训练初始化**：

```python
# 主训练函数
@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    # 1. 动态修改配置
    dynamically_modify_train_config(config)
    
    # 2. 解析配置
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    
    # 3. 可重现性设置
    seed = config.reproduce.seed_everything
    if seed is not None:
        pl.seed_everything(seed=seed, workers=True)
    
    # 4. DDP设置
    strategy = DDPStrategy(process_group_backend='nccl', 
                          find_unused_parameters=False,
                          gradient_as_bucket_view=True) if len(gpus) > 1 else None
    
    # 5. 数据模块
    data_module = fetch_data_module(config=config)
    
    # 6. 日志和检查点
    logger = get_wandb_logger(config)
    ckpt_path = detect_ckpt(ckpt_dir)
    
    # 7. 模型
    module = fetch_model_module(config=config)
    
    # 8. 回调
    callbacks = [
        get_ckpt_callback(config, ckpt_dir=ckpt_dir),
        GradFlowLogCallback(config.logging.train.log_model_every_n_steps * 100),
        LearningRateMonitor(logging_interval='step'),
        get_viz_callback(config=config),
        ModelSummary(max_depth=2)
    ]
    
    # 9. 训练器
    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=20000,
        devices=len(gpus),
        gradient_clip_val=1.0,
        max_steps=400000,
        strategy=strategy,
    )
    
    # 10. 启动训练
    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)
```

### 2. 单步训练循环

#### 混合采样批次处理

```python
def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
    # 1. 合并混合批次
    batch = merge_mixed_batches(batch)
    
    # 2. 提取数据
    data = self.get_data_from_batch(batch)
    worker_id = self.get_worker_id_from_batch(batch)
    
    # 3. 设置模式
    mode = Mode.TRAIN
    ev_tensor_sequence = data[DataType.EV_REPR]
    sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
    is_first_sample = data[DataType.IS_FIRST_SAMPLE]
    
    # 4. 重置RNN状态
    self.mode_2_rnn_states[mode].reset(
        worker_id=worker_id, 
        indices_or_bool_tensor=is_first_sample)
    
    # 5. 处理序列
    L = len(ev_tensor_sequence)
    B = len(sparse_obj_labels[0])
    
    prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
    
    # 6. 时序处理
    for tidx in range(L):
        ev_tensors = ev_tensor_sequence[tidx]
        
        # 主干网络前向传播
        backbone_features, states = self.mdl.forward_backbone(
            x=ev_tensors, 
            previous_states=prev_states)
        
        prev_states = states
        
        # 存储特征和标签
        current_labels, valid_batch_indices = 
            sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
        
        if len(current_labels) > 0:
            backbone_feature_selector.add_backbone_features(
                backbone_features=backbone_features,
                selected_indices=valid_batch_indices)
            obj_labels.extend(current_labels)
    
    # 7. 保存RNN状态
    self.mode_2_rnn_states[mode].save_states_and_detach(
        worker_id=worker_id, states=prev_states)
    
    # 8. 检测头前向传播
    selected_backbone_features = 
        backbone_feature_selector.get_batched_backbone_features()
    
    labels_yolox = ObjectLabels.get_labels_as_batched_tensor(
        obj_label_list=obj_labels, format_='yolox')
    
    predictions, losses = self.mdl.forward_detect(
        backbone_features=selected_backbone_features, 
        targets=labels_yolox)
    
    # 9. 处理混合采样
    if self.mode_2_sampling_mode[mode] in 
            (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
        predictions = predictions[-B:]
        obj_labels = obj_labels[-B:]
    
    # 10. 日志记录
    output = {'loss': losses['loss']}
    self.log_dict({f'{prefix}{k}': v for k, v in losses.items()}, 
                  on_step=True, on_epoch=True, batch_size=B)
    
    return output
```

#### 事件表示填充和归一化

```python
def get_data_from_batch(self, batch: Any):
    data = batch[DATA_KEY]
    
    # 填充事件表示到期望的HxW
    ev_repr = torch.stack(data[DataType.EV_REPR]).to(dtype=self.dtype)
    # [L, B, C, H, W], 事件表示
    
    padded_ev_repr = self.input_padder.pad_tensor_ev_repr(ev_repr)
    data[DataType.EV_REPR] = [ev for ev in padded_ev_repr]  # 回到列表
    
    # 标签子采样以加速训练
    if self.training:
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        for tidx in range(len(sparse_obj_labels)):
            if tidx in self.label_subsample_idx:
                continue
            # 将标签设置为None，除非它是GT标签
            sparse_obj_labels[tidx].set_non_gt_labels_to_none_()
        data[DataType.OBJLABELS_SEQ] = sparse_obj_labels
    
    return data
```

#### 主干网络推理

```python
# 主干网络前向传播
backbone_features, states = self.mdl.forward_backbone(
    x=ev_tensors,
    previous_states=prev_states,
    token_mask=token_masks)

# backbone_features: Dict{stage_id: feats, [B, C, h, w]}
# states: List[(lstm_h, lstm_c), same shape]

# 更新RNN状态
prev_states = states
```

#### 检测头推理

```python
# 准备检测输入
selected_backbone_features = 
    backbone_feature_selector.get_batched_backbone_features()

labels_yolox = ObjectLabels.get_labels_as_batched_tensor(
    obj_label_list=obj_labels, format_='yolox')

# 检测头前向传播
predictions, losses = self.mdl.forward_detect(
    backbone_features=selected_backbone_features, 
    targets=labels_yolox)

# predictions: (B, N, 4 + 1 + num_cls), [(x, y, w, h), obj_conf, cls]
# losses: Dict{loss_name: loss, torch.scalar tensor}
```

#### 损失函数计算

```python
# 损失计算
output = {'loss': losses['loss']}

# 日志记录
prefix = f'{mode_2_string[mode]}/'
log_dict = {f'{prefix}{k}': v for k, v in losses.items()}

self.log_dict(
    log_dict,
    on_step=True,
    on_epoch=True,
    batch_size=B,
    sync_dist=False,
    rank_zero_only=True)
```

#### 流程图和代码示例

```
┌───────────────────────────────────────────────────────┐
│              单步训练循环流程图                        │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              1. 合并混合批次                           │
│  batch = merge_mixed_batches(batch)                   │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              2. 数据预处理                             │
│  - 填充事件表示                                      │
│  - 标签子采样                                        │
│  - 提取worker ID                                    │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              3. RNN状态重置                            │
│  mode_2_rnn_states[mode].reset(worker_id, is_first)   │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              4. 时序处理循环                           │
│  for tidx in range(L):                               │
│    - 提取事件张量                                    │
│    - 主干网络前向传播                                │
│    - 更新RNN状态                                    │
│    - 存储特征和标签                                  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              5. 保存RNN状态                           │
│  mode_2_rnn_states[mode].save_states_and_detach()     │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              6. 检测头前向传播                         │
│  - 准备批次特征                                      │
│  - 转换标签格式                                      │
│  - 检测头推理                                        │
│  - 损失计算                                          │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              7. 处理混合采样                           │
│  - 截取最后批次                                      │
│  - 更新标签                                          │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              8. 日志记录和输出                         │
│  - 记录损失                                          │
│  - 记录指标                                          │
│  - 返回输出                                          │
└───────────────────────────────────────────────────────┘
```

### 3. 验证流程

#### 流式评估机制

```python
def _val_test_step_impl(self, batch: Any, mode: Mode) -> STEP_OUTPUT:
    # 1. 提取数据
    data = self.get_data_from_batch(batch)
    worker_id = self.get_worker_id_from_batch(batch)
    
    # 2. 设置模式
    assert mode in (Mode.VAL, Mode.TEST)
    assert self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM
    
    # 3. 重置RNN状态
    self.mode_2_rnn_states[mode].reset(
        worker_id=worker_id, 
        indices_or_bool_tensor=is_first_sample)
    
    # 4. 时序处理
    for tidx in range(L):
        ev_tensors = ev_tensor_sequence[tidx]
        
        # 主干网络前向传播
        backbone_features, states = self.mdl.forward_backbone(
            x=ev_tensors, previous_states=prev_states)
        
        prev_states = states
        
        # 存储特征和标签
        current_labels, valid_batch_indices = 
            sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
        
        if len(current_labels) > 0:
            backbone_feature_selector.add_backbone_features(
                backbone_features=backbone_features,
                selected_indices=valid_batch_indices)
            obj_labels.extend(current_labels)
    
    # 5. 保存RNN状态
    self.mode_2_rnn_states[mode].save_states_and_detach(
        worker_id=worker_id, states=prev_states)
    
    # 6. 检测头前向传播
    selected_backbone_features = 
        backbone_feature_selector.get_batched_backbone_features()
    
    predictions, _ = self.mdl.forward_detect(
        backbone_features=selected_backbone_features)
    
    # 7. 后处理
    pred_processed = postprocess(
        prediction=predictions,
        num_classes=self.mdl_config.head.num_classes,
        conf_thre=self.mdl_config.postprocess.confidence_threshold,
        nms_thre=self.mdl_config.postprocess.nms_threshold)
    
    # 8. 转换为Prophesee格式
    loaded_labels_proph, yolox_preds_proph = 
        to_prophesee(obj_labels, pred_processed)
    
    # 9. 准备输出
    output = {
        ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-1],
        ObjDetOutput.PRED_PROPH: yolox_preds_proph[-1],
        ObjDetOutput.EV_REPR: ev_repr_selector.get_ev_repr_as_list(start_idx=-1)[0],
        ObjDetOutput.SKIP_VIZ: False,
    }
    
    # 10. 评估
    if self.started_training:
        self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
        self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
    
    return output
```

#### RNN状态保持

```python
# RNN状态管理
self.mode_2_rnn_states[mode].reset(
    worker_id=worker_id, 
    indices_or_bool_tensor=is_first_sample)

# 获取之前的状态
prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)

# 更新状态
prev_states = states

# 保存状态
self.mode_2_rnn_states[mode].save_states_and_detach(
    worker_id=worker_id, states=prev_states)
```

#### Prophesee AP指标计算

```python
def run_psee_evaluator(self, mode: Mode, log: bool = True,
                      reset_buffer: bool = True,
                      ret_pr_curve: bool = False):
    psee_evaluator = self.mode_2_psee_evaluator[mode]
    
    if psee_evaluator.has_data():
        metrics = psee_evaluator.evaluate_buffer(
            img_height=hw_tuple[0],
            img_width=hw_tuple[1],
            ret_pr_curve=ret_pr_curve)
        
        if reset_buffer:
            psee_evaluator.reset_buffer()
        
        # 记录指标
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                value = torch.tensor(v)
            elif isinstance(v, np.ndarray):
                value = torch.from_numpy(v)
            elif isinstance(v, torch.Tensor):
                value = v
            log_dict[f'{prefix}{k}'] = value.to(self.device)
        
        if log:
            self.log_dict(
                log_dict,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True)
```

## 第五部分：配置系统 (Hydra)

### 1. 配置层次结构

```
config/
├── general.yaml             # 通用训练配置
├── dataset/                 # 数据集配置
│   ├── gen1.yaml            # Gen1数据集
│   ├── gen4.yaml            # Gen4数据集
│   └── gen1x0.01_ss.yaml    # 子采样数据集
├── experiment/              # 实验预设
│   ├── gen1/                # Gen1实验
│   │   └── small.yaml       # 小模型配置
│   └── gen4/                # Gen4实验
│       └── small.yaml       # 小模型配置
└── model/                   # 模型配置
    ├── rnndet.yaml          # RNN检测器
    └── pseudo_labeler.yaml  # 伪标签生成器
```

#### general.yaml 通用配置说明

```yaml
reproduce:
  seed_everything: null       # 随机种子
  deterministic_flag: False  # 完全确定性行为
  benchmark: True            # 性能优化
training:
  precision: 16              # 混合精度训练
  max_epochs: 10000          # 最大epoch数
  max_steps: 400000          # 最大步数
  learning_rate: 0.0002      # 学习率
  weight_decay: 0            # 权重衰减
  gradient_clip_val: 1.0     # 梯度裁剪
batch_size:
  train: 8                   # 训练批大小
  eval: 8                    # 评估批大小
hardware:
  num_workers:               # 数据加载worker数
    train: 8
    eval: 8
  gpus: 0                    # GPU设备
  dist_backend: "nccl"       # 分布式后端
```

#### model/ 模型配置

```yaml
# rnndet.yaml
name: rnndet
backbone:
  vit_size: small            # ViT尺寸
  compile:
    enable: True             # torch.compile优化
fpn:
  in_stages: [1, 2, 3, 4]   # 输入阶段
head:
  num_classes: 2            # 类别数
  ignore_label: 1024        # 忽略标签
postprocess:
  confidence_threshold: 0.01 # 置信度阈值
  nms_threshold: 0.45       # NMS阈值
```

#### dataset/ 数据集配置

```yaml
# gen1.yaml
name: gen1
path: ./datasets/gen1/
train:
  sampling: MIXED           # 混合采样
  mixed:
    w_random: 1.0            # 随机采样权重
    w_stream: 1.0            # 流式采样权重
eval:
  sampling: STREAM           # 流式采样
sequence_length: 10         # 序列长度
downsample_by_factor_2: True # 下采样
```

#### experiment/ 实验预设

```yaml
# experiment/gen1/small.yaml
model:
  backbone:
    vit_size: small
  head:
    num_classes: 2
training:
  learning_rate: 0.0002
  max_steps: 200000
```

### 2. 关键配置参数详解

#### 训练参数

| 参数 | 描述 | 默认值 | 范围 |
|------|------|--------|------|
| `learning_rate` | 学习率 | 0.0002 | [1e-5, 1e-3] |
| `max_steps` | 最大训练步数 | 400000 | [10000, 1000000] |
| `batch_size.train` | 训练批大小 | 8 | [1, 32] |
| `gradient_clip_val` | 梯度裁剪值 | 1.0 | [0.1, 10.0] |
| `precision` | 训练精度 | 16 | [16, 32] |

#### 硬件配置

| 参数 | 描述 | 默认值 | 选项 |
|------|------|--------|------|
| `gpus` | GPU设备 | 0 | [0, 1, [0,1], ...] |
| `dist_backend` | 分布式后端 | "nccl" | ["nccl", "gloo"] |
| `num_workers.train` | 训练worker数 | 8 | [0, 16] |
| `num_workers.eval` | 评估worker数 | 8 | [0, 16] |

#### 数据集配置

| 参数 | 描述 | 默认值 | 选项 |
|------|------|--------|------|
| `train.sampling` | 训练采样模式 | MIXED | [RANDOM, STREAM, MIXED] |
| `train.mixed.w_random` | 随机采样权重 | 1.0 | [0.1, 10.0] |
| `train.mixed.w_stream` | 流式采样权重 | 1.0 | [0.1, 10.0] |
| `sequence_length` | 事件序列长度 | 10 | [5, 20] |
| `downsample_by_factor_2` | 是否下采样 | True | [True, False] |

#### 日志配置

| 参数 | 描述 | 默认值 | 选项 |
|------|------|--------|------|
| `wandb.project_name` | WandB项目名 | RVT | 字符串 |
| `logging.train.log_every_n_steps` | 日志记录间隔 | 100 | [10, 1000] |
| `logging.train.high_dim.enable` | 高维可视化 | True | [True, False] |
| `logging.train.high_dim.every_n_steps` | 可视化间隔 | 5000 | [1000, 20000] |

## 第六部分：关键特性和机制

### 1. 事件表示处理

#### 从RAW事件到张量的转换流程

```
┌───────────────────────────────────────────────────────┐
│              RAW事件数据                              │
│  - 时间戳                                          │
│  - 像素坐标 (x, y)                                 │
│  - 极性 (on/off)                                    │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              事件表示构建                            │
│  - 直方图表示                                      │
│  - 混合密度表示                                    │
│  - 时空体素                                        │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              张量转换                                │
│  - 归一化                                          │
│  - 填充到固定尺寸                                  │
│  - 批处理                                          │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              模型输入                                │
│  - [B, C, H, W] 张量                               │
│  - 多通道事件表示                                  │
│  - 时序连续性                                      │
└───────────────────────────────────────────────────────┘
```

#### 直方图表示 vs 混合密度表示

| 特性 | 直方图表示 | 混合密度表示 |
|------|----------|--------------|
| **时间分辨率** | 低 | 高 |
| **空间分辨率** | 高 | 中 |
| **计算效率** | 高 | 中 |
| **内存使用** | 低 | 中 |
| **适用场景** | 快速预览 | 精确检测 |

#### 空间增强策略

```python
# 数据增强
1. **水平翻转**：`th.flip(ev_repr, dims=[-1])`
2. **时间反转**：反转事件序列顺序
3. **随机裁剪**：随机裁剪事件表示
4. **颜色抖动**：调整事件表示强度
5. **混合**：混合多个事件序列
```

#### 填充和归一化

```python
# 填充到固定尺寸
class InputPadderFromShape:
    def __init__(self, desired_hw):
        self.desired_hw = desired_hw
    
    def pad_tensor_ev_repr(self, ev_repr):
        # 填充到 [B, C, H, W]
        padded = pad(ev_repr, (0, 0, 0, 0, 0, 0, 0, self.desired_hw[0] - ev_repr.shape[-2], 
                                    0, self.desired_hw[1] - ev_repr.shape[-1]))
        return padded

# 归一化
# 事件表示通常归一化到 [0, 1] 或 [-1, 1] 范围
# 具体取决于表示类型和数据集
```

### 2. RNN状态管理

#### RNNStates容器设计

```python
class RNNStates:
    """管理多个worker的RNN状态。"""
    
    def __init__(self):
        self.worker_id_2_states = {}  # worker_id -> List[(h, c)]
        self.worker_id_2_device = {}  # worker_id -> device
    
    def reset(self, worker_id, indices_or_bool_tensor):
        """重置特定worker的RNN状态"""
        if worker_id not in self.worker_id_2_states:
            return
        
        # 仅重置选定的样本
        for idx, (h, c) in enumerate(self.worker_id_2_states[worker_id]):
            if indices_or_bool_tensor[idx]:
                h.zero_()
                c.zero_()
    
    def get_states(self, worker_id):
        """获取worker的RNN状态"""
        if worker_id not in self.worker_id_2_states:
            return None
        return self.worker_id_2_states[worker_id]
    
    def save_states_and_detach(self, worker_id, states):
        """保存并分离RNN状态"""
        device = states[0][0].device
        self.worker_id_2_device[worker_id] = device
        self.worker_id_2_states[worker_id] = [
            (h.detach(), c.detach()) for h, c in states
        ]
```

#### 多阶段状态保存

```python
# 多阶段RNN状态管理
1. **每个worker独立**：每个数据加载worker维护独立的RNN状态
2. **按需重置**：根据 `is_first_sample` 标志重置状态
3. **跨批次连续性**：在批次之间保持状态连续性
4. **设备管理**：自动处理设备分配和状态迁移

# 状态流
┌───────────────────────────────────────────────────────┐
│              Batch N                                  │
│  - 提取worker ID                                    │
│  - 检查is_first_sample                               │
│  - 重置状态 (如果需要)                              │
│  - 获取之前状态                                     │
│  - 前向传播                                         │
│  - 更新状态                                         │
│  - 保存并分离状态                                   │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              Batch N+1                                │
│  - 提取worker ID                                    │
│  - 检查is_first_sample                               │
│  - 重置状态 (如果需要)                              │
│  - 获取之前状态 (来自Batch N)                       │
│  - 前向传播                                         │
│  - 更新状态                                         │
│  - 保存并分离状态                                   │
└───────────────────────────────────────────────────────┘
```

#### 状态重置和转移

```python
# 状态重置逻辑
def reset(self, worker_id, indices_or_bool_tensor):
    """重置特定worker的RNN状态"""
    if worker_id not in self.worker_id_2_states:
        return
    
    # 仅重置选定的样本
    for idx, (h, c) in enumerate(self.worker_id_2_states[worker_id]):
        if indices_or_bool_tensor[idx]:
            h.zero_()
            c.zero_()

# 状态转移逻辑
def save_states_and_detach(self, worker_id, states):
    """保存并分离RNN状态"""
    device = states[0][0].device
    self.worker_id_2_device[worker_id] = device
    self.worker_id_2_states[worker_id] = [
        (h.detach(), c.detach()) for h, c in states
    ]
```

#### 跨批次状态连续性

```python
# 跨批次状态连续性机制
1. **worker ID跟踪**：每个批次包含worker ID信息
2. **状态字典**：`worker_id_2_states` 映射worker ID到RNN状态
3. **自动检索**：在每个批次开始时自动检索之前的状态
4. **显式保存**：在每个批次结束时显式保存状态
5. **设备管理**：自动处理设备分配和状态迁移

# 示例流程
batch_1 = loader.next()  # worker_id=0
states_1 = model.forward(batch_1, previous_states=None)
model.save_states(worker_id=0, states=states_1)

batch_2 = loader.next()  # worker_id=0
states_2 = model.forward(batch_2, previous_states=states_1)
model.save_states(worker_id=0, states=states_2)
```

### 3. 伪标签质量控制

#### 低置信度过滤

```python
def filter_pred_boxes(pred_labels, obj_thresh, cls_thresh):
    """过滤低置信度预测"""
    # 计算综合置信度
    conf = pred_labels.objectness * pred_labels.class_confidence
    
    # 应用阈值
    keep = conf >= (obj_thresh * cls_thresh)
    
    # 过滤边界框
    return pred_labels[keep]
```

#### NMS后处理

```python
def tta_postprocess(preds, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    """应用NMS到预测边界框"""
    # 置信度过滤
    conf_mask = ((obj_conf * class_conf) >= conf_thre)
    detections = detections[conf_mask]
    
    # 应用NMS
    if class_agnostic:
        nms_out_index = ops.nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            nms_thre)
    else:
        nms_out_index = ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre)
    
    # 返回过滤后的检测
    return detections[nms_out_index]
```

#### 追踪过滤机制

```python
class LinearTracker:
    """线性追踪器，用于过滤短轨迹和插值补全缺失检测"""
    
    def __init__(self, img_hw, min_hits=3, iou_threshold=0.5):
        self.img_hw = img_hw
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.prev_trackers = []
    
    def update(self, frame_idx, dets, is_gt):
        """更新追踪器状态"""
        # 匹配检测到现有轨迹
        # 创建新轨迹
        # 更新轨迹状态
    
    def get_bbox_tracker(self, bbox_idx):
        """获取特定边界框的追踪器"""
        return self.trackers[bbox_idx]
    
    def finish(self):
        """完成追踪，处理未完成轨迹"""
        # 移动活动轨迹到prev_trackers
        # 标记轨迹为完成

# 追踪过滤逻辑
def _track_filter(self, min_track_len=6, inpaint=False):
    """过滤短轨迹"""
    # 前向追踪
    remove_idx, inpainted_bbox = self._track(
        self.labels, self.frame_idx, 
        min_track_len=min_track_len, inpaint=inpaint)
    
    # 后向追踪
    if 'backward' in track_method:
        rev_labels = [label.get_reverse() for label in self.labels[::-1]]
        rev_frame_idx = [max(self.frame_idx) - idx for idx in self.frame_idx[::-1]]
        bg_remove_idx, _ = self._track(rev_labels, rev_frame_idx, min_track_len=min_track_len)
        
    # 移除短轨迹检测
    for idx, obj_label in enumerate(self.labels):
        for i in range(len(obj_label)):
            if bbox_idx in remove_idx:
                new_class_id[i] = self.filter_config.ignore_label
```

#### 精度和召回率评估

```python
def evaluate_label(pred_labels, gt_labels, obj_thresh, cls_thresh):
    """评估伪标签质量"""
    # 过滤低置信度预测
    pred_labels = filter_pred_boxes(pred_labels, obj_thresh, cls_thresh)
    
    # 计算IoU
    ious = box_iou(pred_labels.bboxes, gt_labels.bboxes)
    
    # 匹配预测和GT
    matches = match_predictions(pred_labels, gt_labels, ious, iou_threshold=0.5)
    
    # 计算TP/FP/FN
    tp = matches.sum()
    fp = len(pred_labels) - tp
    fn = len(gt_labels) - tp
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
```

### 4. 混合采样策略

#### 随机访问 vs 流式加载的权衡

| 方面 | 随机访问 | 流式加载 |
|------|----------|----------|
| **数据多样性** | 高 | 低 |
| **时序连续性** | 无 | 有 |
| **训练速度** | 快 | 慢 |
| **内存使用** | 低 | 高 |
| **数据增强** | 支持 | 不支持 |
| **适用场景** | 预训练 | 微调、评估 |

#### 动态worker分配算法

```python
def set_mixed_sampling_mode_variables_for_train(self):
    """动态分配混合采样资源"""
    # 1. 根据权重设置批大小
    bs_rnd = min(round(self.overall_batch_size_train * weight_random / (weight_stream + weight_random)),
                  self.overall_batch_size_train - 1)
    bs_str = self.overall_batch_size_train - bs_rnd
    
    # 2. 根据批大小设置worker数量
    workers_rnd = min(math.ceil(self.overall_num_workers_train * bs_rnd / self.overall_batch_size_train),
                       self.overall_num_workers_train - 1)
    workers_str = self.overall_num_workers_train - workers_rnd
    
    # 3. 打印配置
    print(f'[Train] Local batch size for:\nstream sampling:\t{bs_str}\nrandom sampling:\t{bs_rnd}\n'
          f'[Train] Local num workers for:\nstream sampling:\t{workers_str}\nrandom sampling:\t{workers_rnd}')
```

#### 批大小和worker数量的配置

```yaml
# 混合采样配置
train:
  sampling: MIXED
  mixed:
    w_random: 1.0      # 随机采样权重
    w_stream: 1.0      # 流式采样权重

hardware:
  num_workers:
    train: 8           # 总worker数
batch_size:
  train: 8            # 总批大小

# 计算结果
# bs_rnd = round(8 * 1.0 / (1.0 + 1.0)) = 4
# bs_str = 8 - 4 = 4
# workers_rnd = ceil(8 * 4 / 8) = 4
# workers_str = 8 - 4 = 4
```

## 第七部分：推理和伪标签生成流程

### 1. 推理流程 (predict.py)

#### 模型加载

```python
# 加载模型
module = fetch_model_module(config=config)

# 加载检查点
if config.checkpoint:
    ckpt_path = config.checkpoint
    print(f'Loading checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    module.load_state_dict(ckpt['state_dict'])
    print(f'Successfully loaded checkpoint')

# 设置为评估模式
module.eval()
```

#### 序列级推理

```python
def predict_sequence(model, dataloader, device):
    """序列级推理"""
    
    # 初始化RNN状态
    rnn_states = None
    
    # 遍历序列
    for batch in dataloader:
        # 移动到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # 前向传播
        with torch.no_grad():
            predictions, _, rnn_states = model(
                event_tensor=batch['event_repr'],
                previous_states=rnn_states,
                retrieve_detections=True)
        
        # 后处理
        pred_processed = postprocess(
            prediction=predictions,
            num_classes=model.num_classes,
            conf_thre=config.confidence_threshold,
            nms_thre=config.nms_threshold)
        
        # 收集结果
        results.append(pred_processed)
    
    return results
```

#### TTA（测试时增强）

```python
# TTA配置
tta_config = config.tta

if tta_config.enable:
    print('Using TTA in inference')
    
    # 水平翻转增强
    if tta_config.hflip:
        hflip_results = []
        for batch in dataloader:
            hflip_batch = {k: th.flip(v, dims=[-1]) if k == 'event_repr' else v 
                          for k, v in batch.items()}
            
            with torch.no_grad():
                hflip_pred, _, _ = model(
                    event_tensor=hflip_batch['event_repr'],
                    previous_states=None,
                    retrieve_detections=True)
            
            hflip_pred = postprocess(hflip_pred, ...)
            hflip_results.append(hflip_pred)
        
        # 合并结果
        results = ensemble_results(original_results, hflip_results)
    
    # 时间反转增强
    if tta_config.tflip:
        # 反转事件序列
        # 推理
        # 合并结果
        pass
```

#### 数据格式转换

```python
# 转换为Prophesee格式
def to_prophesee(obj_labels, pred_processed):
    """转换为Prophesee评估格式"""
    
    # 准备标签
    loaded_labels_proph = []
    for obj_label in obj_labels:
        if obj_label is not None and len(obj_label) > 0:
            label_proph = obj_label.get_labels_as_tensors(format_='prophesee')
            loaded_labels_proph.append(label_proph)
    
    # 准备预测
    yolox_preds_proph = []
    for pred in pred_processed:
        if pred is not None and len(pred) > 0:
            pred_proph = convert_yolox_to_prophesee(pred)
            yolox_preds_proph.append(pred_proph)
    
    return loaded_labels_proph, yolox_preds_proph
```

### 2. 伪标签生成步骤

#### 完整序列推理

```python
def generate_pseudo_labels(model, dataloader, config):
    """生成伪标签"""
    
    # 初始化
    ev_path_2_ev_data = {}
    
    # 遍历数据集
    for batch in dataloader:
        # 提取数据
        data = batch[DATA_KEY]
        ev_paths = data[DataType.PATH]
        
        # 推理
        with torch.no_grad():
            predictions, _, _ = model(
                event_tensor=data[DataType.EV_REPR],
                previous_states=None,
                retrieve_detections=True)
        
        # 后处理
        pred_processed = postprocess(predictions, ...)
        
        # 转换为ObjectLabels
        pred_labels = pred2label(pred_processed, ...)
        
        # 收集数据
        for i, (path, pred_label) in enumerate(zip(ev_paths, pred_labels)):
            if path not in ev_path_2_ev_data:
                ev_path_2_ev_data[path] = EventSeqData(path, ...)
            
            ev_path_2_ev_data[path].update(
                labels=[pred_label],
                ev_idx=data[DataType.EV_IDX][i],
                is_last_sample=data[DataType.IS_LAST_SAMPLE][i],
                is_padded_mask=data[DataType.IS_PADDED_MASK][i],
                is_hflip=False,
                is_tflip=False,
                tflip_offset=0)
    
    return ev_path_2_ev_data
```

#### NMS和置信度过滤

```python
# 置信度过滤
obj_thresh = config.pseudo_label.obj_thresh
cls_thresh = config.pseudo_label.cls_thresh

for path, ev_data in ev_path_2_ev_data.items():
    # 过滤低置信度预测
    filtered_labels = []
    for label in ev_data.labels:
        if label is not None:
            filtered = filter_pred_boxes(label, obj_thresh, cls_thresh)
            filtered_labels.append(filtered)
    
    ev_data.labels = filtered_labels
```

#### 线性追踪器应用

```python
# 应用追踪过滤
for path, ev_data in ev_path_2_ev_data.items():
    if len(ev_data.labels) > 0:
        # 前向追踪
        remove_idx, inpainted_bbox = ev_data._track(
            ev_data.labels,
            ev_data.frame_idx,
            min_track_len=config.filter.min_track_len,
            inpaint=config.filter.inpaint)
        
        # 后向追踪
        if 'backward' in config.filter.track_method:
            rev_labels = [label.get_reverse() for label in ev_data.labels[::-1]]
            rev_frame_idx = [max(ev_data.frame_idx) - idx for idx in ev_data.frame_idx[::-1]]
            bg_remove_idx, _ = ev_data._track(rev_labels, rev_frame_idx, 
                                            min_track_len=config.filter.min_track_len,
                                            inpaint=False)
            
        # 移除短轨迹检测
        ev_data._apply_remove_indices(remove_idx)
        
        # 插值补全
        if inpainted_bbox:
            ev_data._apply_inpainted_bbox(inpainted_bbox)
```

#### 伪标签保存格式

```python
# 保存伪标签
def save_pseudo_labels(ev_path_2_ev_data, save_dir):
    """保存伪标签到磁盘"""
    
    for path, ev_data in ev_path_2_ev_data.items():
        # 创建目录结构
        new_seq_dir = os.path.join(save_dir, os.path.basename(path))
        os.makedirs(new_seq_dir, exist_ok=True)
        
        # 软链接事件数据
        ev_h5_fn = get_ev_h5_fn(path)
        new_ev_h5_fn = get_ev_h5_fn(new_seq_dir)
        os.symlink(ev_h5_fn, new_ev_h5_fn)
        
        # 处理和收集标签
        ev_data._aggregate_results()
        ev_data._track_filter()
        labels, objframe_idx_2_label_idx, objframe_idx_2_repr_idx = ev_data._summarize()
        
        # 保存索引映射
        np.save(os.path.join(new_seq_dir, 'objframe_idx_2_repr_idx.npy'), 
                objframe_idx_2_repr_idx)
        
        # 保存标签
        np.savez(
            os.path.join(new_seq_dir, 'labels.npz'),
            labels=labels,
            objframe_idx_2_label_idx=objframe_idx_2_label_idx)
        
        # 链接验证/测试集
        link_val_test_sets(path, new_seq_dir)
```

#### 质量验证步骤

```python
def validate_pseudo_labels(config):
    """验证伪标签质量"""
    
    # 加载伪标签数据集
    pseudo_dataset = PseudoLabelDataset(config.save_dir)
    
    # 加载模型
    model = load_model(config.checkpoint)
    
    # 评估质量
    metrics = evaluate_dataset(model, pseudo_dataset)
    
    # 打印结果
    print(f'Pseudo label quality metrics:')
    print(f'  Precision: {metrics["precision"]:.4f}')
    print(f'  Recall: {metrics["recall"]:.4f}')
    print(f'  AP: {metrics["ap"]:.4f}')
    
    # 检查一致性
    check_consistency(pseudo_dataset)
    
    return metrics
```

### 3. 自训练循环

#### 第一轮预训练

```bash
# Gen1预训练 (1% 数据)
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.01_ss \
  +experiment/gen1="small.yaml" training.max_steps=200000

# Gen4预训练 (1% 数据)
python train.py model=rnndet hardware.gpus=[0,1] dataset=gen4x0.01_ss \
  +experiment/gen4="small.yaml" training.max_steps=200000
```

#### 伪标签生成和验证

```bash
# 生成伪标签
python predict.py model=pseudo_labeler dataset=gen1x0.01_ss dataset.path=./datasets/gen1/ \
  checkpoint="pretrained/gen1x0.01_ss.ckpt" hardware.gpus=0 +experiment/gen1="small.yaml" \
  model.postprocess.confidence_threshold=0.01 tta.enable=True \
  save_dir=./datasets/pseudo_gen1/gen1x0.01_ss-1round/train

# 验证伪标签质量
python val_dst.py model=pseudo_labeler dataset=gen1x0.01_ss \
  dataset.path=./datasets/pseudo_gen1/gen1x0.01_ss-1round \
  checkpoint=1 +experiment/gen1="small.yaml" \
  model.pseudo_label.obj_thresh=0.01 model.pseudo_label.cls_thresh=0.01
```

#### 软目标分配

```python
# 软目标分配配置
model:
  head:
    use_soft_labels: True      # 启用软标签
    soft_label_weight: 0.5    # 软标签权重
    temperature: 2.0          # 温度参数

# 训练配置
training:
  learning_rate: 0.0005      # 更大的学习率
  max_steps: 150000          # 更少的步数
```

#### 多轮迭代策略

```bash
# 第一轮自训练
python train.py model=rnndet-soft hardware.gpus=0 dataset=gen1x0.01_ss-1round \
  +experiment/gen1="small.yaml" training.max_steps=150000 training.learning_rate=0.0005

# 第二轮伪标签生成
python predict.py model=pseudo_labeler dataset=gen1x0.01_ss-1round dataset.path=./datasets/pseudo_gen1/gen1x0.01_ss-1round/ \
  checkpoint="pretrained/gen1x0.01_ss-1round.ckpt" hardware.gpus=0 +experiment/gen1="small.yaml" \
  model.postprocess.confidence_threshold=0.01 tta.enable=True \
  save_dir=./datasets/pseudo_gen1/gen1x0.01_ss-2round/train

# 第二轮自训练
python train.py model=rnndet-soft hardware.gpus=0 dataset=gen1x0.01_ss-2round \
  +experiment/gen1="small.yaml" training.max_steps=150000 training.learning_rate=0.0005
```

## 第八部分：性能优化

### 1. 内存优化

#### 梯度累积

```python
# 梯度累积配置
training:
  accumulate_grad_batches: 4  # 梯度累积批次数
  batch_size: 8              # 基础批大小

# 实际批大小 = batch_size * accumulate_grad_batches = 32

# 实现
optimizer.zero_grad()
for i in range(accumulate_grad_batches):
    batch = next(dataloader)
    loss = model(batch)
    loss.backward()
optimizer.step()
```

#### 混合精度训练（FP16）

```python
# 混合精度配置
training:
  precision: 16  # 16位混合精度

# PyTorch Lightning自动处理
# - 自动混合精度 (AMP)
# - 梯度缩放
# - FP16计算 + FP32权重更新

# 手动实现
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    with torch.cuda.amp.autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### TorchData流式加载

```python
# TorchData优化
1. **内存映射**：直接从磁盘加载数据
2. **异步加载**：后台加载数据
3. **批处理**：优化批处理
4. **缓存**：缓存常用数据

# 实现
from torchdata.datapipes import iter

datapipe = iter.FileLister(root=dataset_path)
    .open_files(mode='b')
    .load_from_tar()
    .shuffle()
    .batch(batch_size)
    .collate()
    .prefetch(num_workers)
```

#### 动态批大小

```python
# 动态批大小配置
training:
  dynamic_batch_size: True
  base_batch_size: 8
  max_batch_size: 32
  scale_factor: 1.5

# 实现
current_batch_size = base_batch_size

for epoch in range(num_epochs):
    try:
        batch = next(dataloader)
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # 增加批大小
        if epoch % 10 == 0 and current_batch_size < max_batch_size:
            current_batch_size = min(int(current_batch_size * scale_factor), max_batch_size)
            dataloader.batch_size = current_batch_size
    except RuntimeError as e:
        if "out of memory" in str(e):
            # 减少批大小
            current_batch_size = max(int(current_batch_size / scale_factor), base_batch_size)
            dataloader.batch_size = current_batch_size
```

### 2. 计算优化

#### torch.compile 图编译

```python
# torch.compile配置
model:
  backbone:
    compile:
      enable: True      # 启用编译
      mode: "max-autotune"  # 编译模式

# 实现
if self.mdl_config.backbone.compile.enable:
    self.mdl.backbone = torch.compile(
        self.mdl.backbone,
        mode=self.mdl_config.backbone.compile.mode)

# 编译模式选项
# - "default": 平衡优化
# - "reduce-overhead": 减少开销
# - "max-autotune": 最大自动调优
# - "max-autotune-no-cudagraphs": 无CUDA图的最大自动调优
```

#### CudaTimer 性能分析

```python
# 性能分析
from utils.timers import CudaTimer

# 使用示例
with CudaTimer(device=x.device, timer_name="Backbone"):
    backbone_features, states = self.backbone(x, previous_states, token_mask)

with CudaTimer(device=device, timer_name="FPN"):
    fpn_features = self.fpn(backbone_features)

with CudaTimer(device=device, timer_name="HEAD + Loss"):
    outputs, losses = self.yolox_head(fpn_features, targets, soft_targets)

# 输出分析
CudaTimer.print_summary()
```

#### DDP 分布式加速

```python
# DDP配置
strategy = DDPStrategy(
    process_group_backend='nccl',
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    bucket_cap_mb=25,
    static_graph=True)

# 关键参数
# - process_group_backend: 通信后端 (nccl, gloo)
# - find_unused_parameters: 是否查找未使用参数
# - gradient_as_bucket_view: 梯度桶视图
# - bucket_cap_mb: 桶大小 (MB)
# - static_graph: 静态图优化

# 优化技巧
1. **桶大小调整**：根据网络带宽调整
2. **梯度压缩**：启用梯度压缩
3. **重叠计算和通信**：异步通信
4. **静态图**：减少开销
```

#### 多worker并行加载

```python
# 多worker配置
hardware:
  num_workers:
    train: 8
    eval: 8

# 优化技巧
1. **worker初始化**：避免GIL竞争
2. **数据加载优化**：
   - 使用内存映射
   - 批处理I/O操作
   - 缓存常用数据
3. **worker分配**：
   - 随机采样：更多worker
   - 流式采样：更少worker
4. **内存管理**：
   - 限制worker内存
   - 使用共享内存

# 实现
DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2)
```

### 3. 集群适配 (Slurm)

#### 自动检查点恢复

```python
# 检查点恢复逻辑
def detect_ckpt(ckpt_path: str):
    """自动检测和恢复检查点"""
    last_ckpt = None
    
    if os.path.exists(ckpt_path):
        ckp_files = glob_all(ckpt_path)
        ckp_files = [ckp for ckp in ckp_files if ckp.endswith('.ckpt')]
        
        if ckp_files:
            ckp_files = sort_file_by_time(ckp_files)
            last_ckpt = ckp_files[-1]
            
            try:
                _ = torch.load(last_ckpt, map_location='cpu')
            except:
                os.remove(last_ckpt)
                last_ckpt = None
                if len(ckp_files) > 1:
                    last_ckpt = ckp_files[-2]
            
            print(f'INFO: automatically detect checkpoint {last_ckpt}')
    
    return last_ckpt
```

#### 软链接临时存储

```python
# 软链接优化
if SLURM_JOB_ID and os.path.isdir('/checkpoint/'):
    usr = pwd.getpwuid(os.getuid())[0]
    new_dir = f'/checkpoint/{usr}/{SLURM_JOB_ID}/'
    
    # 检查点目录可能已存在
    if os.path.exists(ckpt_dir):
        old_slurm_id = find_old_slurm_id(ckpt_dir)
        
        if old_slurm_id is None:
            slurm_id = SLURM_JOB_ID
        
        wandb_name = f'{exp_name}-{slurm_id}'
        
        # 移动文件到新目录
        if str(old_slurm_id) != str(SLURM_JOB_ID):
            for f in sort_file_by_time(glob_all(ckpt_dir)):
                if 'SLURM_JOB_FINISHED' in f:
                    os.system(f'rm -f {f}')
                else:
                    os.system(f'mv {f} {new_dir}')
        
        os.system(f'rm -rf {ckpt_dir}')
    
    # 创建软链接
    os.system(f'ln -s {new_dir} {ckpt_dir}')
    os.system(f"touch {os.path.join(ckpt_dir, 'DELAYPURGE')}")
```

#### 预留处理和任务重启

```python
# 预留处理
1. **检查点保护**：
   - 创建DELAYPURGE文件
   - 防止自动清除
   
2. **任务重启**：
   - 检测旧SLURM ID
   - 恢复WandB运行
   - 继续训练

# 实现
# 检查预留
if os.path.exists(ckpt_dir):
    old_slurm_id = find_old_slurm_id(ckpt_dir)
    
    if old_slurm_id and str(old_slurm_id) != str(SLURM_JOB_ID):
        # 恢复旧任务
        wandb_name = f'{exp_name}-{old_slurm_id}'
        
        # 移动文件
        for f in glob_all(ckpt_dir):
            os.system(f'mv {f} {new_dir}')
        
        os.system(f'rm -rf {ckpt_dir}')
        os.system(f'ln -s {new_dir} {ckpt_dir}')
```

#### WandB运行ID保持

```python
# WandB运行ID保持
if os.path.exists(ckpt_dir):
    old_slurm_id = find_old_slurm_id(ckpt_dir)
    
    if old_slurm_id:
        slurm_id = old_slurm_id
    else:
        slurm_id = SLURM_JOB_ID
    
    wandb_name = wandb_id = f'{exp_name}-{slurm_id}'
    config.wandb.wandb_name = wandb_name
    config.wandb.wandb_id = wandb_id
    
    # 恢复WandB运行
    logger = get_wandb_logger(config)
    
    # 继续训练
    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)
```

## 第九部分：完整训练示例

### 1. 基础模型预训练（1%/2%/5%数据）

#### Gen1预训练 (1% 数据)

```bash
# 单GPU训练
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.01_ss \
  +experiment/gen1="small.yaml" training.max_steps=200000

# 配置详情
# - model: rnndet (RNN检测器)
# - dataset: gen1x0.01_ss (Gen1, 1% 数据, 半监督)
# - experiment: small.yaml (小模型配置)
# - training.max_steps: 200000 (20万步)

# 期望输出
# - 检查点保存到: ./checkpoint/rvt-s-gen1x0.01_ss-bs8_iter200k/
# - WandB日志: rvt-s-gen1x0.01_ss-bs8_iter200k-{SLURM_JOB_ID}
# - 训练时间: ~12小时 (单GPU)
```

#### Gen4预训练 (1% 数据)

```bash
# 双GPU训练
python train.py model=rnndet hardware.gpus=[0,1] dataset=gen4x0.01_ss \
  +experiment/gen4="small.yaml" training.max_steps=200000

# 配置详情
# - hardware.gpus: [0,1] (双GPU)
# - dataset: gen4x0.01_ss (Gen4, 1% 数据, 半监督)
# - batch_size.train: 8 (每GPU)
# - 实际批大小: 16 (8 * 2)

# 期望输出
# - 检查点保存到: ./checkpoint/rvt-s-gen4x0.01_ss-bs16_iter200k/
# - 训练时间: ~8小时 (双GPU)
```

#### 不同数据比例的训练步数

```yaml
# Appendix A.2 - 不同数据比例的训练步数
# 数据比例 | 训练步数 | 学习率      | 批大小
# 1%       | 200k     | 0.0002      | 8/16
# 2%       | 300k     | 0.0002      | 8/16
# 5%       | 400k     | 0.0002      | 8/16
# 10%+     | 400k     | 0.0002      | 8/16

# Gen1命令
python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.02_ss \
  +experiment/gen1="small.yaml" training.max_steps=300000

python train.py model=rnndet hardware.gpus=0 dataset=gen1x0.05_ss \
  +experiment/gen1="small.yaml" training.max_steps=400000
```

### 2. 伪标签生成和验证

#### Gen1伪标签生成

```bash
# 生成伪标签 (单GPU, ~7小时)
python predict.py model=pseudo_labeler dataset=gen1x0.01_ss dataset.path=./datasets/gen1/ \
  checkpoint="pretrained/gen1x0.01_ss.ckpt" hardware.gpus=0 +experiment/gen1="small.yaml" \
  model.postprocess.confidence_threshold=0.01 tta.enable=True \
  save_dir=./datasets/pseudo_gen1/gen1x0.01_ss-1round/train

# 参数说明
# - model: pseudo_labeler (伪标签生成器)
# - checkpoint: 预训练模型路径
# - confidence_threshold: 0.01 (低阈值以保留更多预测)
# - tta.enable: True (启用测试时增强)
# - save_dir: 保存路径

# 输出结构
# ./datasets/pseudo_gen1/gen1x0.01_ss-1round/
# ├── train/
# │   ├── 18-03-29_13-15-02_500000_60500000/
# │   │   ├── event_representations_v2/
# │   │   │   └── ev_representation_name/
# │   │   │       ├── event_representations.h5 (软链接)
# │   │   │       └── objframe_idx_2_repr_idx.npy
# │   │   └── labels_v2/
# │   │       └── labels.npz
# │   ├── ... (其他序列)
# ├── val/ (软链接到原始val集)
# └── test/ (软链接到原始test集)
```

#### Gen4伪标签生成

```bash
# 生成伪标签 (单GPU, ~10小时)
python predict.py model=pseudo_labeler dataset=gen4x0.01_ss dataset.path=./datasets/gen4/ \
  checkpoint="pretrained/gen4x0.01_ss.ckpt" hardware.gpus=0 +experiment/gen4="small.yaml" \
  model.postprocess.confidence_threshold=0.01 tta.enable=True \
  save_dir=./datasets/pseudo_gen4/gen4x0.01_ss-1round/train

# 注意: Gen4数据集更大，需要更多时间
# 输出大小: 200-250 MB
# 总迭代次数: 27044
```

#### 伪标签质量验证

```bash
# 验证Gen1伪标签质量
python val_dst.py model=pseudo_labeler dataset=gen1x0.01_ss \
  dataset.path=./datasets/pseudo_gen1/gen1x0.01_ss-1round \
  checkpoint=1 +experiment/gen1="small.yaml" \
  model.pseudo_label.obj_thresh=0.01 model.pseudo_label.cls_thresh=0.01

# 参数说明
# - checkpoint=1: 使用伪标签 (不是模型检查点)
# - obj_thresh: 0.01 (对象置信度阈值)
# - cls_thresh: 0.01 (类别置信度阈值)

# 期望输出
# - 精度 (Precision): ~0.7-0.9
# - 召回率 (Recall): ~0.6-0.8
# - AP: ~0.5-0.7

# Gen4验证
python val_dst.py model=pseudo_labeler dataset=gen4x0.01_ss \
  dataset.path=./datasets/pseudo_gen4/gen4x0.01_ss-1round \
  checkpoint=1 +experiment/gen4="small.yaml" \
  model.pseudo_label.obj_thresh=0.01 model.pseudo_label.cls_thresh=0.01
```

### 3. 自训练（软锚点分配）

#### 第一轮自训练

```bash
# Gen1自训练 (单GPU)
python train.py model=rnndet-soft hardware.gpus=0 dataset=gen1x0.01_ss-1round \
  +experiment/gen1="small.yaml" training.max_steps=150000 training.learning_rate=0.0005

# 关键差异
# - model: rnndet-soft (启用软锚点分配)
# - dataset: gen1x0.01_ss-1round (伪标签数据集)
# - training.max_steps: 150000 (更少步数)
# - training.learning_rate: 0.0005 (更大学习率)

# 原因
# - 更密集的注释 (伪标签)
# - 更快的收敛
# - 更大的有效批大小

# Gen4自训练 (双GPU)
python train.py model=rnndet-soft hardware.gpus=[0,1] dataset=gen4x0.01_ss-1round \
  +experiment/gen4="small.yaml" training.max_steps=150000 training.learning_rate=0.0005
```

#### 第二轮伪标签生成

```bash
# 基于第一轮自训练模型生成新伪标签
python predict.py model=pseudo_labeler dataset=gen1x0.01_ss-1round \
  dataset.path=./datasets/pseudo_gen1/gen1x0.01_ss-1round/ \
  checkpoint="pretrained/gen1x0.01_ss-1round.ckpt" hardware.gpus=0 \
  +experiment/gen1="small.yaml" model.postprocess.confidence_threshold=0.01 \
  tta.enable=True save_dir=./datasets/pseudo_gen1/gen1x0.01_ss-2round/train

# 注意: 使用第一轮自训练模型作为检查点
# 期望质量提升: AP提高 ~2-5%
```

#### 第二轮自训练

```bash
# 基于第二轮伪标签进行自训练
python train.py model=rnndet-soft hardware.gpus=0 dataset=gen1x0.01_ss-2round \
  +experiment/gen1="small.yaml" training.max_steps=150000 training.learning_rate=0.0005

# 期望性能: 接近饱和，进一步提升有限
# 根据论文Sec.4.4，性能增益在第二轮后减少
```

### 4. 最终模型评估

#### 标准评估

```bash
# Gen1评估
python val.py model=rnndet dataset=gen1 dataset.path=./datasets/gen1/ \
  checkpoint="pretrained/gen1x0.01_ss-2round.ckpt" use_test_set=1 \
  hardware.gpus=0 hardware.num_workers.eval=8 +experiment/gen1="small.yaml" \
  batch_size.eval=16 model.postprocess.confidence_threshold=0.001 \
  reverse=False tta.enable=False

# 参数说明
# - use_test_set=1: 使用测试集
# - confidence_threshold=0.001: 低阈值以获取最佳mAP
# - reverse=False: 不反转时间顺序
# - tta.enable=False: 不启用TTA

# 期望输出
# - mAP: ~40-50% (取决于数据比例)
# - 评估时间: ~1小时

# Gen4评估
python val.py model=rnndet dataset=gen4 dataset.path=./datasets/gen4/ \
  checkpoint="pretrained/gen4x0.01_ss-2round.ckpt" use_test_set=1 \
  hardware.gpus=0 hardware.num_workers.eval=8 +experiment/gen4="small.yaml" \
  batch_size.eval=8 model.postprocess.confidence_threshold=0.001 \
  reverse=False tta.enable=False
```

#### 带TTA评估

```bash
# 带TTA的Gen1评估
python val.py model=rnndet dataset=gen1 dataset.path=./datasets/gen1/ \
  checkpoint="pretrained/gen1x0.01_ss-2round.ckpt" use_test_set=1 \
  hardware.gpus=0 hardware.num_workers.eval=8 +experiment/gen1="small.yaml" \
  batch_size.eval=16 model.postprocess.confidence_threshold=0.001 \
  reverse=False tta.enable=True

# TTA影响
# - 评估时间: ~2小时 (2x慢)
# - mAP提升: ~1-3%
# - 更稳健的检测

# 注意: TTA在论文中用于伪标签生成，但不一定用于最终评估
```

#### 反向时间评估

```bash
# 反向时间评估
python val.py model=rnndet dataset=gen1 dataset.path=./datasets/gen1/ \
  checkpoint="pretrained/gen1x0.01_ss-2round.ckpt" use_test_set=1 \
  hardware.gpus=0 hardware.num_workers.eval=8 +experiment/gen1="small.yaml" \
  batch_size.eval=16 model.postprocess.confidence_threshold=0.001 \
  reverse=True tta.enable=False

# 反向评估目的
# - 测试模型对时间方向的鲁棒性
# - 评估RNN状态管理的有效性
# - 发现潜在的时间偏见

# 期望结果
# - mAP下降: ~5-10% (正常)
# - 如果下降过大 (>20%)，可能表明时间偏见
```

## 第十部分：关键创新点总结

### 1. 标签高效学习框架

**创新点**：
- **混合采样策略**：结合随机访问和流式采样，平衡效率和时序连续性
- **自训练框架**：迭代伪标签生成和模型训练
- **软锚点分配**：改进的锚点分配策略，适用于稀疏标签

**优势**：
- 减少对密集标注的依赖
- 提高小数据集上的性能
- 加速收敛

**代码实现**：
```python
# 混合采样
self.train_sampling_mode = DatasetSamplingMode.MIXED

# 自训练
python predict.py model=pseudo_labeler ...  # 生成伪标签
python train.py model=rnndet-soft ...       # 软锚点训练

# 软锚点
model:
  head:
    use_soft_labels: True
```

### 2. 混合采样策略

**创新点**：
- **动态权重分配**：根据配置自动分配批大小和worker数量
- **RNN状态管理**：为每种采样模式独立管理RNN状态
- **批次合并**：无缝合并来自不同采样模式的批次

**优势**：
- 平衡训练效率和时序连续性
- 优化资源利用
- 支持灵活的采样比例

**代码实现**：
```python
def set_mixed_sampling_mode_variables_for_train(self):
    # 动态分配资源
    bs_rnd = min(round(self.overall_batch_size_train * weight_random / (weight_stream + weight_random)),
                  self.overall_batch_size_train - 1)
    bs_str = self.overall_batch_size_train - bs_rnd

def training_step(self, batch):
    batch = merge_mixed_batches(batch)  # 合并批次
```

### 3. RNN状态连续性管理

**创新点**：
- **跨批次状态保持**：在批次之间保持RNN状态
- **worker级状态管理**：为每个数据加载worker独立管理状态
- **自动重置**：根据`is_first_sample`标志自动重置状态

**优势**：
- 保持事件序列的时序连续性
- 支持流式数据加载
- 优化内存使用

**代码实现**：
```python
class RNNStates:
    def save_states_and_detach(self, worker_id, states):
        # 保存并分离状态
        self.worker_id_2_states[worker_id] = [
            (h.detach(), c.detach()) for h, c in states
        ]
    
    def get_states(self, worker_id):
        # 获取之前的状态
        return self.worker_id_2_states.get(worker_id, None)
```

### 4. 伪标签质量控制

**创新点**：
- **双向追踪过滤**：前向和后向追踪以过滤短轨迹
- **插值补全**：补全缺失检测以提高连续性
- **综合置信度过滤**：结合对象和类别置信度

**优势**：
- 提高伪标签质量
- 减少噪声标签
- 改进模型收敛

**代码实现**：
```python
def _track_filter(self):
    # 双向追踪
    remove_idx, inpainted_bbox = self._track(self.labels, self.frame_idx, 
                                           min_track_len=6, inpaint=True)
    
    # 后向追踪
    if 'backward' in track_method:
        rev_labels = [label.get_reverse() for label in self.labels[::-1]]
        bg_remove_idx, _ = self._track(rev_labels, rev_frame_idx, min_track_len=6)
        
    # 移除短轨迹
    for idx, obj_label in enumerate(self.labels):
        for i in range(len(obj_label)):
            if bbox_idx in remove_idx:
                new_class_id[i] = self.filter_config.ignore_label
```

### 5. 集群系统优化

**创新点**：
- **自动检查点恢复**：在SLURM抢占系统中自动检测和恢复检查点
- **软链接临时存储**：优化集群存储使用
- **WandB运行保持**：在任务重启时保持WandB运行ID

**优势**：
- 提高集群利用率
- 减少存储成本
- 支持长时间运行任务

**代码实现**：
```python
# 自动检查点恢复
def detect_ckpt(ckpt_path):
    if os.path.exists(ckpt_path):
        ckp_files = glob_all(ckpt_path)
        ckp_files = [ckp for ckp in ckp_files if ckp.endswith('.ckpt')]
        if ckp_files:
            last_ckpt = sort_file_by_time(ckp_files)[-1]
            return last_ckpt

# 软链接优化
if SLURM_JOB_ID and os.path.isdir('/checkpoint/'):
    os.system(f'ln -s {new_dir} {ckpt_dir}')
    os.system(f"touch {os.path.join(ckpt_dir, 'DELAYPURGE')}")
```

## 第十一部分：依赖关系和模块交互

### 训练流程的模块交互图

```
┌───────────────────────────────────────────────────────┐
│                   train.py (主入口)                    │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              Hydra配置加载                            │
│  - 加载general.yaml                                  │
│  - 合并实验配置                                      │
│  - 解析参数                                          │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              数据模块初始化                           │
│  modules/data/genx.py                                │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ DataModule.setup()│    │ 构建数据集            │  │
│  │ - 设置采样模式    │    │ - 随机访问数据集      │  │
│  │ - 分配worker     │    │ - 流式数据集          │  │
│  │ - 创建数据加载器 │    │ - 混合采样配置        │  │
│  └─────────────────┘    └─────────────────────────┘  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              模型初始化                               │
│  modules/detection.py                                │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ Module.__init__()│    │ YoloXDetector          │  │
│  │ - 加载配置        │    │ - MaxViT-RNN骨干       │  │
│  │ - 创建输入填充器 │    │ - PAFPN特征融合       │  │
│  │ - 初始化RNN状态  │    │ - YOLOX检测头         │  │
│  └─────────────────┘    └─────────────────────────┘  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              训练循环                                 │
│  pl.Trainer.fit()                                    │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 1. 数据加载      │    │ 2. 前向传播            │  │
│  │ - 合并混合批次    │    │ - 主干网络            │  │
│  │ - 提取数据       │    │ - 特征融合            │  │
│  │ - 获取worker ID  │    │ - 检测头              │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 3. RNN状态管理  │    │ 4. 损失计算            │  │
│  │ - 重置状态       │    │ - 计算损失            │  │
│  │ - 获取之前状态   │    │ - 后处理              │  │
│  │ - 保存新状态     │    │ - 日志记录            │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐                                │
│  │ 5. 反向传播      │                                │
│  │ - 梯度计算      │                                │
│  │ - 优化器步骤    │                                │
│  └─────────────────┘                                │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              验证和检查点                             │
│  - 定期验证                                        │
│  - 保存检查点                                      │
│  - 日志指标                                        │
│  - 可视化                                          │
└───────────────────────────────────────────────────────┘
```

### 伪标签生成的完整流程

```
┌───────────────────────────────────────────────────────┐
│              predict.py (伪标签生成入口)              │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              模型加载                                │
│  - 加载检查点                                      │
│  - 设置为评估模式                                  │
│  - 初始化TTA配置                                    │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              数据加载                                │
│  modules/data/genx.py                              │
│  - 构建流式数据集                                  │
│  - 单序列加载                                      │
│  - 无数据增强                                      │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              序列级推理                              │
│  modules/pseudo_labeler.py                          │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 1. 初始化        │    │ 2. 时序处理          │  │
│  │ - 创建EventSeqData│    │ - 遍历序列          │  │
│  │ - 设置过滤配置    │    │ - 前向传播          │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 3. 后处理        │    │ 4. 收集结果          │  │
│  │ - NMS            │    │ - 按路径分组        │  │
│  │ - 置信度过滤     │    │ - 存储预测          │  │
│  └─────────────────┘    └─────────────────────────┘  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              追踪和过滤                              │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 1. 双向追踪      │    │ 2. 过滤短轨迹        │  │
│  │ - 前向追踪       │    │ - 移除噪声检测      │  │
│  │ - 后向追踪       │    │ - 保留高质量检测    │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐                                │
│  │ 3. 插值补全      │                                │
│  │ - 补全缺失检测  │                                │
│  │ - 提高连续性    │                                │
│  └─────────────────┘                                │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              保存伪标签                              │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 1. 创建目录      │    │ 2. 软链接事件数据    │  │
│  │ - 目录结构       │    │ - 避免数据复制      │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 3. 保存标签      │    │ 4. 保存索引映射      │  │
│  │ - labels.npz     │    │ - objframe_idx_2_*  │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐                                │
│  │ 5. 链接val/test │                                │
│  │ - 软链接         │                                │
│  └─────────────────┘                                │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│              质量验证                                │
│  - 计算精度/召回率                                  │
│  - 计算AP指标                                      │
│  - 检查一致性                                      │
│  - 输出报告                                        │
└───────────────────────────────────────────────────────┘
```

### 数据流向说明

```
┌───────────────────────────────────────────────────────┐
│                   原始数据集                          │
│  ./datasets/gen1/                                   │
│  ├── train/                                         │
│  │   ├── 18-03-29_13-15-02_500000_60500000/           │
│  │   │   ├── event_representations_v2/               │
│  │   │   │   └── ev_representation_name/             │
│  │   │   │       ├── event_representations.h5        │
│  │   │   │       └── objframe_idx_2_repr_idx.npy     │
│  │   │   └── labels_v2/                             │
│  │   │       └── labels.npz                         │
│  │   └── ... (其他序列)                              │
│  ├── val/                                           │
│  └── test/                                          │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│                   数据加载器                          │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 随机访问加载器   │    │ 流式加载器             │  │
│  │ - 批处理         │    │ - 单序列              │  │
│  │ - 打乱           │    │ - 顺序加载            │  │
│  │ - 数据增强       │    │ - 无增强              │  │
│  └─────────────────┘    └─────────────────────────┘  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│                   模型输入                            │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 事件表示         │    │ 标签                  │  │
│  │ - [B, C, H, W]   │    │ - ObjectLabels        │  │
│  │ - 归一化         │    │ - 稀疏格式            │  │
│  │ - 填充           │    │ - 批处理              │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐                                │
│  │ RNN状态         │                                │
│  │ - worker ID     │                                │
│  │ - 是否首帧      │                                │
│  └─────────────────┘                                │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│                   模型输出                            │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 检测结果         │    │ RNN状态               │  │
│  │ - 边界框         │    │ - 更新状态            │  │
│  │ - 置信度         │    │ - 分离以备后用        │  │
│  │ - 类别           │                                │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐                                │
│  │ 损失            │                                │
│  │ - 分类损失      │                                │
│  │ - 定位损失      │                                │
│  │ - 对象损失      │                                │
│  └─────────────────┘                                │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│                   伪标签生成                          │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 原始预测        │    │ 过滤后预测            │  │
│  │ - 所有检测       │    │ - 高置信度            │  │
│  │ - 低阈值        │    │ - NMS后处理          │  │
│  └─────────────────┘    └─────────────────────────┘  │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 追踪过滤         │    │ 最终伪标签            │  │
│  │ - 短轨迹移除     │    │ - 高质量            │  │
│  │ - 插值补全       │    │ - 连续              │  │
│  └─────────────────┘    └─────────────────────────┘  │
└───────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────┐
│                   伪标签数据集                        │
│  ./datasets/pseudo_gen1/gen1x0.01_ss-1round/          │
│  ├── train/                                         │
│  │   ├── 18-03-29_13-15-02_500000_60500000/           │
│  │   │   ├── event_representations_v2/               │
│  │   │   │   └── ev_representation_name/             │
│  │   │   │       ├── event_representations.h5 (链接) │
│  │   │   │       └── objframe_idx_2_repr_idx.npy     │
│  │   │   └── labels_v2/                             │
│  │   │       └── labels.npz (伪标签)                 │
│  │   └── ... (其他序列)                              │
│  ├── val/ (链接到原始)                               │
│  └── test/ (链接到原始)                              │
└───────────────────────────────────────────────────────┘
```

## 第十二部分：常见问题和调试技巧

### 1. 内存不足处理

#### 症状
- `CUDA out of memory` 错误
- 训练中途崩溃
- 显存使用迅速增加

#### 解决方案

```python
# 1. 减小批大小
batch_size:
  train: 4  # 从8减少到4

# 2. 启用梯度累积
training:
  accumulate_grad_batches: 2  # 实际批大小 = 4 * 2 = 8

# 3. 减少序列长度
dataset:
  sequence_length: 5  # 从10减少到5

# 4. 禁用torch.compile
model:
  backbone:
    compile:
      enable: False

# 5. 减少worker数量
hardware:
  num_workers:
    train: 4  # 从8减少到4

# 6. 启用梯度检查点
model:
  backbone:
    use_gradient_checkpointing: True
```

#### 调试命令

```bash
# 监控GPU内存
watch -n 1 nvidia-smi

# 使用更小的批大小测试
python train.py ... batch_size.train=2

# 启用内存分析
CUDA_LAUNCH_BLOCKING=1 python train.py ...
```

### 2. 模型加载错误

#### 症状
- 形状不匹配错误
- 缺失键错误
- 版本不兼容错误

#### 解决方案

```python
# 1. 检查模型尺寸
# 错误: 尝试加载RVT-B权重到RVT-S模型
# 解决: 确保模型尺寸匹配

# 2. 检查数据集
# 错误: 在Gen1上训练的模型在Gen4上评估
# 解决: 确保数据集匹配

# 3. 检查实验配置
# 错误: 使用错误的实验配置
# 解决: 检查 +experiment/gen1="small.yaml"

# 4. 部分加载
# 如果不兼容，尝试部分加载
model.load_state_dict(ckpt['state_dict'], strict=False)
```

#### 调试命令

```bash
# 检查检查点内容
python -c "import torch; ckpt = torch.load('xxx.ckpt'); print(ckpt.keys())"

# 检查模型架构
python -c "from models.detection.yolox_extension.models.detector import YoloXDetector; import torch; model = YoloXDetector(...); print(model)"

# 比较形状
python -c "import torch; ckpt = torch.load('xxx.ckpt'); model = ...; for k, v in ckpt['state_dict'].items(): print(k, v.shape, getattr(model, k).shape)"
```

### 3. 数据格式问题

#### 症状
- 标签格式错误
- 事件表示不匹配
- 索引超出范围

#### 解决方案

```python
# 1. 验证数据集格式
python val_dst.py ... checkpoint=1

# 2. 检查标签结构
import numpy as np
data = np.load('labels.npz')
print(data['labels'].dtype)  # 应该是BBOX_DTYPE
print(data['objframe_idx_2_label_idx'].shape)

# 3. 验证事件表示
import h5py
with h5py.File('event_representations.h5', 'r') as f:
    print(f['data'].shape)  # 应该是 [N, C, H, W]
    print(f['data'].dtype)  # 应该是float32

# 4. 检查索引映射
repr_idx = np.load('objframe_idx_2_repr_idx.npy')
print(repr_idx.shape)
print(repr_idx.dtype)  # 应该是int64
```

#### 调试命令

```bash
# 验证单个序列
python -c "
from data.genx_utils.labels import ObjectLabels
import numpy as np

# 加载标签
data = np.load('labels.npz')
labels = ObjectLabels(data['labels'], (260, 346))
print('Label count:', len(labels))
print('First label:', labels[0])
"

# 检查事件表示
python -c "
import h5py
with h5py.File('event_representations.h5', 'r') as f:
    print('Shape:', f['data'].shape)
    print('Dtype:', f['data'].dtype)
    print('Min/Max:', f['data'][:].min(), f['data'][:].max())
"
```

### 4. 性能瓶颈分析

#### 症状
- 训练速度慢
- GPU利用率低
- 数据加载缓慢

#### 解决方案

```python
# 1. 分析性能
from utils.timers import CudaTimer

with CudaTimer(device='cuda', timer_name="Total"):
    # 训练循环
    for batch in dataloader:
        with CudaTimer(device='cuda', timer_name="Data Loading"):
            # 数据加载
            pass
        
        with CudaTimer(device='cuda', timer_name="Forward"):
            # 前向传播
            pass
        
        with CudaTimer(device='cuda', timer_name="Backward"):
            # 反向传播
            pass

CudaTimer.print_summary()

# 2. 优化数据加载
hardware:
  num_workers:
    train: 12  # 增加worker数量

dataset:
  train:
    random:
      weighted_sampling: False  # 禁用加权采样

# 3. 优化模型
model:
  backbone:
    compile:
      enable: True
      mode: "max-autotune"

# 4. 混合精度
training:
  precision: 16

# 5. 减少日志频率
logging:
  train:
    log_every_n_steps: 1000  # 从100增加到1000
```

#### 调试命令

```bash
# 监控GPU利用率
watch -n 1 nvidia-smi

# 分析数据加载时间
python -c "
import time
from torch.utils.data import DataLoader
from data.genx_utils.dataset_streaming import build_streaming_dataset

dataset = build_streaming_dataset(...)
loader = DataLoader(dataset, batch_size=None, num_workers=8)

start = time.time()
for i, batch in enumerate(loader):
    if i % 100 == 0:
        print(f'Batch {i}: {time.time() - start:.2f}s')
        start = time.time()
"

# 分析模型性能
python -c "
import torch
from models.detection.yolox_extension.models.detector import YoloXDetector

model = YoloXDetector(...)
model.eval()

# 热身
for _ in range(10):
    x = torch.randn(1, 3, 260, 346).cuda()
    with torch.no_grad():
        model(x)

# 基准测试
import time
start = time.time()
for _ in range(100):
    x = torch.randn(1, 3, 260, 346).cuda()
    with torch.no_grad():
        model(x)
print(f'Average time: {(time.time() - start) / 100 * 1000:.2f}ms')
"
```

### 5. 常见错误消息

#### 错误：`AssertionError: Different GT on the same frame!`

**原因**：相同帧有不同的GT标签

**解决方案**：
```python
# 检查数据集一致性
python val_dst.py ... checkpoint=1

# 如果是伪标签生成中的问题
# 确保 use_gt=True 仅用于Gen1
model:
  use_gt: True  # 仅适用于Gen1
```

#### 错误：`RuntimeError: CUDA out of memory`

**解决方案**：
```bash
# 减小批大小
python train.py ... batch_size.train=4

# 启用梯度累积
python train.py ... training.accumulate_grad_batches=2

# 使用更小的模型
python train.py ... +experiment/gen1="tiny.yaml"
```

#### 错误：`KeyError: 'state_dict'`

**解决方案**：
```bash
# 检查检查点内容
python -c "import torch; ckpt = torch.load('xxx.ckpt'); print(ckpt.keys())"

# 如果是伪标签检查点
python val_dst.py ... checkpoint=1

# 如果是模型检查点
python train.py ... checkpoint="xxx.ckpt"
```

#### 错误：`AssertionError: data/pseudo label formats are wrong`

**解决方案**：
```bash
# 验证数据集格式
python val_dst.py ... checkpoint=1

# 检查数据集路径
# 确保 dataset.path 正确
```

## 结论

本文档提供了LEOD项目的全面架构分析，涵盖了从核心模块设计到训练流程的各个方面。通过详细的代码示例和流程图，开发者可以快速理解项目的关键创新点和实现细节。

### 关键要点总结

1. **混合采样策略**：平衡随机访问和流式加载，优化训练效率和时序连续性
2. **RNN状态管理**：跨批次保持LSTM状态，确保事件序列的连续性
3. **伪标签自训练**：迭代生成和使用伪标签，实现标签高效学习
4. **模型架构**：MaxViT-RNN骨干 + PAFPN + YOLOX检测头
5. **集群优化**：自动检查点恢复和存储管理

### 后续步骤

1. 参考[install.md](install.md)设置环境
2. 参考[benchmark.md](benchmark.md)运行实验
3. 根据本文档理解和修改代码
4. 贡献改进和优化

### 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交拉取请求
4. 遵循代码风格和文档标准
5. 包含测试和基准测试

### 联系方式

如有任何问题，请联系：Ziyi Wu dazitu616@gmail.com

### 许可证

LEOD在MIT许可证下发布。详见LICENSE文件。
