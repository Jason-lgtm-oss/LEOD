# LEOD Implementation Guide
## Deep Dive into Key Modules and Algorithms

---

## Part 1: Recurrent Backbone Implementation

### 1.1 LSTM State Initialization and Management

**File:** `modules/utils/detection.py`

```python
class RNNStates:
    """
    Per-worker LSTM state container for streaming data.
    
    Key insight: In distributed training with multiple workers,
    each worker maintains its own sequence context. We cannot
    mix states between workers.
    """
    
    def __init__(self):
        self.worker_id_2_states: Dict[int, LstmStates] = {}
        # LstmStates = List[(h, c), ...], one per stage
    
    def reset(self, worker_id: int, indices_or_bool_tensor):
        """
        Reset states at sequence boundaries.
        
        Args:
            worker_id: DataLoader worker ID
            indices_or_bool_tensor: 
                - Tensor[B] if streaming: True means new sequence
                - List[int] if selective reset
        
        Example:
            # Streaming batch
            is_first = torch.tensor([True, False, False])  # 3 samples
            rnn_states.reset(worker_id=0, indices_or_bool_tensor=is_first)
            # → States reset for sample 0, preserved for samples 1,2
            
            # Random-access batch
            is_first = torch.tensor([True, True, True])  # All random
            rnn_states.reset(worker_id=0, indices_or_bool_tensor=is_first)
            # → All states reset (expected, no temporal context)
        """
        if worker_id not in self.worker_id_2_states:
            self.worker_id_2_states[worker_id] = None
            return
        
        if isinstance(indices_or_bool_tensor, torch.Tensor):
            if indices_or_bool_tensor.all():
                # Reset all states
                self.worker_id_2_states[worker_id] = None
            else:
                # Partial reset - selective per-sample reset
                # (Advanced case, rarely used)
                current_states = self.worker_id_2_states[worker_id]
                for idx in indices_or_bool_tensor.nonzero():
                    # Reset individual sample's states
                    for stage_idx in range(len(current_states)):
                        h, c = current_states[stage_idx]
                        h[idx] = 0
                        c[idx] = 0
    
    def get_states(self, worker_id: int) -> Optional[LstmStates]:
        """Get previous frame's states, or None if not initialized"""
        return self.worker_id_2_states.get(worker_id, None)
    
    def save_states_and_detach(self, worker_id: int, states: LstmStates):
        """
        Save states for next frame, MUST detach from graph.
        
        Why detach?
        -----------
        Gradients should NOT flow back through state layers.
        If we don't detach:
        
        Frame 1: LSTM hidden state h_1
        Frame 2: LSTM takes h_1, computes h_2
        Frame 3: LSTM takes h_2, computes h_3
        ...
        Frame 100: LSTM takes h_99, computes h_100
        
        Backprop would compute grad flow:
        loss ← h_100 ← h_99 ← ... ← h_1 ← h_0
        
        This causes:
        1. Gradient explosion (BPTT without truncation)
        2. Very large memory (keep all intermediate states)
        3. Training instability
        
        Solution: Detach at batch boundaries
        loss ← h_100 [gradient stops here]
        Previous frames: stopped gradient flow
        
        LSTM will still learn temporal patterns through:
        - Input-to-hidden weights (no detach)
        - Hidden-to-hidden weights within forward pass
        Just not across batch boundaries (which is fine).
        """
        detached_states = []
        for state in states:
            if state is None:
                detached_states.append(None)
            else:
                h, c = state
                # Detach: h_new is computed from h_old
                # But gradients of h_new don't affect h_old
                detached_states.append((h.detach(), c.detach()))
        
        self.worker_id_2_states[worker_id] = detached_states
```

### 1.2 Forward Pass Through Recurrent Stages

**File:** `modules/detection.py`, `training_step()` method

```python
def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
    """Process L frames of event sequence"""
    
    batch = merge_mixed_batches(batch)  # Unify random/streaming batch structure
    data = self.get_data_from_batch(batch)
    worker_id = self.get_worker_id_from_batch(batch)
    
    mode = Mode.TRAIN
    ev_tensor_sequence = data[DataType.EV_REPR]  # List of L tensors
    sparse_obj_labels = data[DataType.OBJLABELS_SEQ]  # List of L label objects
    is_first_sample = data[DataType.IS_FIRST_SAMPLE]  # [B] bool
    
    # 1. RESET RNN STATES AT SEQUENCE BOUNDARIES
    self.mode_2_rnn_states[mode].reset(
        worker_id=worker_id,
        indices_or_bool_tensor=is_first_sample
    )
    
    L = len(ev_tensor_sequence)  # Number of frames (e.g., 10)
    B = len(sparse_obj_labels[0])  # Batch size
    
    # 2. INITIALIZE PREVIOUS STATES
    prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
    # prev_states = None (new sequences) or List[(h,c)] per stage
    
    # 3. SELECT FEATURES AT LABELED FRAMES
    backbone_feature_selector = BackboneFeatureSelector()
    obj_labels = []
    
    # 4. LOOP THROUGH L FRAMES
    for tidx in range(L):
        # Frame at time t
        ev_tensors = ev_tensor_sequence[tidx]  # [B, C, H, W]
        
        # 5. FORWARD BACKBONE
        backbone_features, states = self.mdl.forward_backbone(
            x=ev_tensors,
            previous_states=prev_states,  # ← Key: temporal context
            token_mask=None  # (Would mask padding, not used in training)
        )
        # backbone_features: Dict[int, Tensor[B, C, h, w]]
        #   stage_1: [B, 64, 96, 128]
        #   stage_2: [B, 128, 48, 64]
        #   stage_3: [B, 256, 24, 32]
        #   stage_4: [B, 512, 12, 16]
        #
        # states: List of 4 tuples [(h,c), ...]
        #   Each h,c shape: [B, stage_dim, h_stage, w_stage]
        
        prev_states = states  # ← Save for next frame
        
        # 6. SELECT FEATURES AT LABELED FRAMES ONLY
        current_labels, valid_batch_indices = \
            sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
        
        if len(current_labels) > 0:
            # Only keep features where we have labels
            backbone_feature_selector.add_backbone_features(
                backbone_features=backbone_features,
                selected_indices=valid_batch_indices
            )
            obj_labels.extend(current_labels)
    
    # 7. SAVE FINAL STATES FOR NEXT BATCH
    self.mode_2_rnn_states[mode].save_states_and_detach(
        worker_id=worker_id,
        states=prev_states
    )
    
    # 8. COMPUTE LOSS ON SELECTED FRAMES
    selected_backbone_features = \
        backbone_feature_selector.get_batched_backbone_features()
    
    predictions, losses = self.mdl.forward_detect(
        backbone_features=selected_backbone_features,
        targets=labels_yolox
    )
    
    return {'loss': losses['loss']}
```

**Execution Example:**

```
Streaming batch: L=10 frames, B=2 samples
is_first_sample = [True, False]  # Sample 0 starts new seq, sample 1 continues

Frame 0:
  prev_states = None  # New sequence
  forward_backbone(ev[0], prev_states=None)
  → states_0 = [(h_0, c_0) per stage]
  save prev_states = states_0

Frame 1:
  prev_states = states_0  # Continue from frame 0
  forward_backbone(ev[1], prev_states=states_0)
  → states_1 = [(h_1, c_1) per stage]
  save prev_states = states_1
  Note: h_1 = LSTM(ev[1], h_0)  [temporal context!]

...

Frame 9:
  prev_states = states_8
  forward_backbone(ev[9], prev_states=states_8)
  → states_9 = [(h_9, c_9) per stage]
  save prev_states = states_9

Next batch (same sequence, sample 1):
  is_first_sample = [False, True]  # Sample 0 continues, sample 1 resets
  reset(is_first=[False, True])
  → Sample 0: preserve states_9
  → Sample 1: reset to None
  
  prev_states = get_states() = states_9 (for sample 0)
  forward_backbone(ev_10, prev_states=states_9)
```

### 1.3 Visualizing LSTM State Flow

```
Event Sequence (1000 frames @ 1000 FPS)
Subsampled to 10 consecutive frames for one training batch

Frame Index:  0     1     2     3     4  ...  99    100
Ground Truth: Label None  None  None  None ... None  Label
Event Repr:   [B,2,H,W]

LSTM Forward Pass:
┌────────────────────────────────────────────────────────────┐
│ Time Step 0 (Frame 0)                                      │
│ Input: ev_0[B,2,H,W]                                       │
│ LSTM Input: prev_states = None                             │
│ LSTM Forward: h_0, c_0 = LSTM_cell(ev_0, h=0, c=0)         │
│ Output: features_0, states_0=(h_0, c_0)                    │
│ Save: states_0 → worker[0]                                 │
└────────────────────────────────────────────────────────────┘
           ↓
┌────────────────────────────────────────────────────────────┐
│ Time Step 1 (Frame 1)                                      │
│ Input: ev_1[B,2,H,W]                                       │
│ LSTM Input: prev_states = states_0                         │
│ LSTM Forward: h_1, c_1 = LSTM_cell(ev_1, h_0, c_0)         │
│            ↑                          ↑   ↑                │
│            └──────────────────────────┴───┘  Temporal!    │
│ Output: features_1, states_1=(h_1, c_1)                    │
│ Save: states_1 → worker[0]                                 │
└────────────────────────────────────────────────────────────┘
           ↓
        (repeat for frames 2-9)
           ↓
┌────────────────────────────────────────────────────────────┐
│ Time Step 9 (Frame 9)                                      │
│ Input: ev_9[B,2,H,W]                                       │
│ LSTM Input: prev_states = states_8=(h_8, c_8)              │
│ LSTM Forward: h_9, c_9 = LSTM_cell(ev_9, h_8, c_8)         │
│ Output: features_9, states_9=(h_9, c_9)                    │
│ Save: states_9 → worker[0]                                 │
│ DETACH: states_9.detach()                                  │
│ Gradient flow STOPS here ✓                                 │
└────────────────────────────────────────────────────────────┘
           ↓
   Next batch arrives (same sequence)
           ↓
┌────────────────────────────────────────────────────────────┐
│ Time Step 10 (Frame 10)                                    │
│ Input: ev_10[B,2,H,W]                                      │
│ LSTM Input: prev_states = states_9.detach()                │
│ LSTM Forward: h_10, c_10 = LSTM_cell(ev_10, h_9, c_9)      │
│             ↑                                ↑   ↑          │
│             └────────────────────────────────┴───┘          │
│ Note: h_9 is detached, no gradient to frame 0              │
│ But LSTM still learns from ev_10 and its own weights!      │
└────────────────────────────────────────────────────────────┘
```

---

## Part 2: Pseudo-Label Generation Pipeline

### 2.1 TTA (Test-Time Augmentation) Aggregation

**File:** `modules/pseudo_labeler.py`, lines 37-91

```python
def tta_postprocess(preds: List[ObjectLabels],
                    conf_thre: float = 0.7,
                    nms_thre: float = 0.45,
                    class_agnostic: bool = False) -> List[ObjectLabels]:
    """
    Merge predictions from multiple augmentations via NMS.
    
    Typical flow:
    1. Run model on original image → preds_original
    2. Run model on h-flipped image → preds_hflip (flip bboxes back)
    3. Run model on t-flipped sequence → preds_tflip
    4. Run model on both flips → preds_hflip_tflip
    
    All 4 predictions are on the SAME frames (after flipping back).
    Now merge them with high-precision filtering.
    """
    
    if len(preds) == 0:
        return preds
    
    pad = preds[0].new_zeros()
    output = [pad] * len(preds)
    
    for i, pred in enumerate(preds):
        # Skip processing for ground truth (never filtered)
        if pred.is_gt_label().any():
            output[i] = pred
            continue
        
        # Convert to [(x1,y1,x2,y2), obj_conf, cls_conf, cls_idx] format
        t = pred.t.unsqueeze(1)  # [N, 1]
        pred_tensor = pred.get_labels_as_tensors(format_='prophesee')
        # pred_tensor shape: [N, 7]
        # Fields: [x1, y1, x2, y2, obj_conf, cls_conf, cls_idx]
        
        if not pred_tensor.size(0):
            continue  # Empty frame
        
        # 1. CONFIDENCE FILTERING
        # Key insight: Combine object confidence and class confidence
        # High-confidence detections: both objectness AND class score high
        obj_conf = pred_tensor[:, 4]  # [N]
        class_conf = pred_tensor[:, 5]  # [N]
        combined_conf = obj_conf * class_conf  # [N]
        
        conf_mask = (combined_conf >= conf_thre)  # [N] bool
        detections = pred_tensor[conf_mask]  # Keep only high-conf
        t = t[conf_mask]
        
        if not detections.size(0):
            continue  # All filtered out
        
        # 2. NMS (Non-Maximum Suppression)
        # Remove overlapping detections within same class
        
        if class_agnostic:
            # Single NMS across all classes
            nms_out_index = ops.nms(
                detections[:, :4],  # Box coordinates
                detections[:, 4] * detections[:, 5],  # Combined score
                nms_thre  # IoU threshold
            )
        else:
            # Per-class NMS (keeps non-overlapping boxes in different classes)
            nms_out_index = ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6].long(),  # Class ID for per-class NMS
                nms_thre
            )
        
        detections = detections[nms_out_index]
        t = t[nms_out_index]
        
        # 3. CONVERT BACK TO ObjectLabels FORMAT
        # From (x1,y1,x2,y2) to (x,y,w,h) for consistency
        xywh = xyxy2xywh(detections[:, :4],
                          format_='corner', last4=True)
        obj_conf, class_conf, cls_id = \
            torch.split(detections[:, 4:], 1, dim=1)
        
        # Reconstruct: [t, x, y, w, h, cls_id, cls_conf, obj_conf]
        detections = th.cat([t, xywh, cls_id, class_conf, obj_conf], dim=1)
        output[i] = ObjectLabels(detections, pad.input_size_hw)
    
    return output
```

**Example: Merging TTA Predictions**

```
Frame 10 (with ground truth label)
Original: [bbox_1 (0.95, 0.9), bbox_2 (0.92, 0.88)]
HFlip:    [bbox_1 (0.94, 0.91), bbox_2 (0.90, 0.87)]
TFlip:    [bbox_1 (0.93, 0.92), bbox_2 (0.91, 0.86)]
Combine:  [bbox_1 (0.88, 0.89), bbox_2 (0.85, 0.84)]

All 4 predict same 2 objects ✓
Average confidence: bbox_1 = 0.925, bbox_2 = 0.885
After NMS (IoU 0.45): keep both (non-overlapping)

Frame 11 (no ground truth label, only pseudo-labels)
Original: [bbox_3 (0.8, 0.85), bbox_4 (0.45, 0.50)]
HFlip:    [bbox_3 (0.78, 0.84), bbox_5 (0.35, 0.40)]
TFlip:    [bbox_4 (0.44, 0.48), bbox_5 (0.38, 0.42)]
Combine:  [bbox_6 (0.40, 0.45)]

Inference:
- bbox_3, bbox_4 appear in >1 augmentation → likely true object
- bbox_5, bbox_6 appear in only 1 aug → likely false positive

After TTA NMS with conf_thre=0.7 and strict NMS:
→ Only bbox_3 and bbox_4 survive
→ High-quality pseudo-labels!
```

### 2.2 Linear Tracking for Filtering

**File:** `modules/tracking/linear.py` + `modules/pseudo_labeler.py` lines 201-260

```python
@staticmethod
def _track(labels: List[ObjectLabels],
           frame_idx: List[int],
           min_track_len: int = 6,
           inpaint: bool = False):
    """
    Filter pseudo-labels using linear motion tracking.
    
    Key insight: True objects follow smooth motion
    False positives appear randomly in 1-2 frames
    
    Algorithm:
    1. Track each bbox across frames
    2. Remove tracklets with <min_track_len hits
    3. Hallucinate detections in missed frames of good tracklets
    """
    
    model = LinearTracker(img_hw=labels[0].input_size_hw)
    
    # 1. FEED ALL FRAMES TO TRACKER
    # Some frames have detections, others don't
    # Tracker predicts position for frames without detections
    
    for f_idx in range(max(frame_idx) + 1):
        if f_idx not in frame_idx:
            # No detection at this frame
            # Tracker predicts position based on linear motion
            model.update(f_idx)
            continue
        
        # Frame with detections
        idx = frame_idx.index(f_idx)
        obj_label: ObjectLabels = labels[idx]
        
        # Get bboxes in [x, y, w, h, cls_id] format
        obj_label.numpy_()
        bboxes = obj_label.get_xywh(format_='center', add_class_id=True)
        is_gt = obj_label.is_gt_label()
        
        # Feed to tracker
        model.update(frame_idx=f_idx, dets=bboxes, is_gt=is_gt)
    
    model.finish()
    
    # 2. FILTER TRACKLETS
    # A tracklet = sequence of matched detections for same object
    
    remove_idx = []
    bbox_idx = 0
    
    for obj_label in labels:
        for _ in range(len(obj_label)):
            tracker = model.get_bbox_tracker(bbox_idx)
            
            # Decide: keep or remove this bbox?
            keep = False
            
            if not tracker.done:
                # Tracklet still active (appearing in later frames)
                # → Likely true object
                keep = True
            elif tracker.is_gt:
                # Original ground truth
                # → Always keep
                keep = True
            elif tracker.hits >= min_track_len:
                # Appeared in >= min_track_len frames
                # → Likely true object
                keep = True
            else:
                # Short tracklet (< min_track_len hits)
                # → Likely false positive
                keep = False
            
            if not keep:
                remove_idx.append(bbox_idx)
            
            bbox_idx += 1
    
    # 3. INPAINTING (Optional)
    # For valid tracklets, hallucinate detections at missed frames
    
    inpainted_bbox = {}
    
    if inpaint:
        for tracker in model.prev_trackers:
            # Check if tracklet is "good"
            is_good = (tracker.done and tracker.hits >= min_track_len) or \
                      tracker.is_gt
            
            if not is_good:
                continue
            
            # For this tracklet, look at missed frames
            for f_idx, bbox in tracker.missed_bbox.items():
                # bbox was predicted (not detected) at frame f_idx
                # Add to inpainted_bbox to be added back to labels
                
                if f_idx not in inpainted_bbox:
                    inpainted_bbox[f_idx] = []
                
                inpainted_bbox[f_idx].append(bbox)
    
    return remove_idx, inpainted_bbox
```

**Example: Tracking in Action**

```
Frame Index:  0     1     2     3     4     5     6
Detections:   obj1  obj1  —     —     obj1  —     obj1
              obj2  obj2  obj2  obj2  obj2  —     —

Object 1:
  Frame 0: detect ✓ (initialize tracker, hits=1)
  Frame 1: detect ✓ (hits=2)
  Frame 2: miss   — (predict position)
  Frame 3: miss   — (predict position)
  Frame 4: detect ✓ (hits=3)
  Frame 5: miss   — (predict position)
  Frame 6: detect ✓ (hits=4)
  
  hits=4 >= min_track_len=6? NO
  → REMOVE this tracklet (short, likely FP)

Object 2:
  Frame 0: detect ✓ (initialize tracker, hits=1)
  Frame 1: detect ✓ (hits=2)
  Frame 2: detect ✓ (hits=3)
  Frame 3: detect ✓ (hits=4)
  Frame 4: detect ✓ (hits=5)
  Frame 5: miss   — (predict position, missed_bbox[5])
  Frame 6: miss   — (predict position, missed_bbox[6])
  
  hits=5 < min_track_len=6? Still no...
  But it's still "done"? (will check later)
  → If deemed valid, inpaint detections at frames 5,6

Filtering Decision:
  Remove object 1 (too short, isolated detections)
  Keep object 2 (long continuous track, few misses)
```

### 2.3 EventSeqData - Per-Sequence Aggregation

**File:** `modules/pseudo_labeler.py`, lines 94-200

```python
class EventSeqData:
    """Record labels for a single event sequence"""
    
    def __init__(self, path: str, scale_ratio: int,
                 filter_config: DictConfig, postproc_cfg: DictConfig):
        self.path = path
        self.scale_ratio = scale_ratio  # Downsample ratio during inference
        self.filter_config = filter_config
        self.postproc_cfg = postproc_cfg
        
        # Key data structure
        self.frame_idx_2_labels: Dict[int, ObjectLabels] = {}
        # Maps frame_index → predicted labels
        
        self._eoe = False  # End of sequence flag
        self._aug = False  # Whether TTA augmentations were used
    
    def update(self, labels: List[ObjectLabels], ev_idx: List[int],
               is_last_sample: bool, is_padded_mask: List[bool],
               is_hflip: bool, is_tflip: bool, tflip_offset: int):
        """
        Called once per forward pass with augmentation type.
        Aggregates predictions from all 4 TTA variants.
        
        Args:
            labels: Model predictions for this batch
            ev_idx: Frame indices corresponding to labels
            is_hflip: Whether this pass used horizontal flip
            is_tflip: Whether this pass used temporal flip
            tflip_offset: Offset for temporal flip
                        (number of frames for reversing sequence)
        
        Example flow:
        
        Pass 1 (Original):
          ev_idx = [10, 11, 12, 13]  (frame indices)
          labels = [Label(bboxes), Label(bboxes), ...]
          is_hflip=False, is_tflip=False
          → Store at frame_idx_2_labels[10], [11], [12], [13]
        
        Pass 2 (HFlip):
          ev_idx = [10, 11, 12, 13]  (same frames, flipped)
          labels = [Label(flipped bboxes), ...]  (bboxes flipped back)
          is_hflip=True, is_tflip=False
          → Merge with frame_idx_2_labels[10], [11], [12], [13]
        
        Pass 3 (TFlip):
          ev_idx = [103, 102, 101, 100]  (reversed: 100+3, 100+2, ...)
          tflip_offset = 104
          Modified ev_idx = [100, 101, 102, 103]  (flip back)
          → Merge with frame_idx_2_labels[100], [101], [102], [103]
        
        Pass 4 (HFlip + TFlip):
          ev_idx = [103, 102, 101, 100]
          Modified → [100, 101, 102, 103]
          → Merge with frame_idx_2_labels[100], [101], [102], [103]
        """
        self._eoe = is_last_sample
        
        # Reverse frame indices if temporal flip
        if is_tflip:
            ev_idx = [i + tflip_offset for i in ev_idx]
        
        # Apply horizontal flip to bboxes if needed
        if is_hflip:
            labels = self._hflip_bbox(labels)
            self._aug = True
        
        # Record that we used augmentations
        if is_tflip:
            self._aug = True
        
        # Update internal dictionary
        self._update(labels, ev_idx, is_padded_mask)
    
    def _update(self, labels: List[ObjectLabels], ev_idx: List[int],
                is_padded_mask: List[bool]):
        """Merge labels into frame_idx_2_labels"""
        
        for tidx, (label, frame_idx) in enumerate(zip(labels, ev_idx)):
            if frame_idx < 0 or label is None or len(label) == 0:
                continue
            
            if is_padded_mask[tidx]:
                continue  # Skip padding
            
            # Scale back to original resolution
            label.scale_(self.scale_ratio)
            
            if frame_idx in self.frame_idx_2_labels:
                # Check for GT label duplication
                if label.is_gt_label().any():
                    # GT labels should match exactly
                    try:
                        assert label == self.frame_idx_2_labels[frame_idx], \
                            'Different GT on same frame!'
                    except AssertionError:
                        # Allowed small numerical differences
                        pass
                    continue  # Don't re-add GT
                
                # Merge predicted labels via += operator
                self.frame_idx_2_labels[frame_idx] += label
            else:
                # First time seeing this frame
                self.frame_idx_2_labels[frame_idx] = label
    
    def _aggregate_results(self, num_frames: int):
        """After sequence ends, merge TTA and apply post-processing"""
        assert self._eoe, 'Sequence not finished'
        
        # Convert to sorted lists
        if len(self.frame_idx_2_labels) == 0:
            self.frame_idx, self.labels = [], []
            return
        
        frame_idx = sorted([i for i in self.frame_idx_2_labels.keys()
                           if 0 <= i < num_frames])
        self.frame_idx = frame_idx
        self.labels = [self.frame_idx_2_labels[idx] for idx in frame_idx]
        
        # If TTA was used, apply NMS to merge predictions
        if self._aug:
            self.labels = tta_postprocess(
                self.labels,
                conf_thre=self.postproc_cfg.confidence_threshold,
                nms_thre=self.postproc_cfg.nms_threshold
            )
```

---

## Part 3: Data Mixing Strategy

### 3.1 Mixed Dataloader Implementation

**File:** `modules/data/genx.py`

```python
def get_dataloader_kwargs(dataset, sampling_mode, dataset_mode,
                          dataset_config, batch_size, num_workers):
    """
    Create DataLoader kwargs based on sampling mode.
    
    Key differences:
    - Streaming: batch_size=None (entire sequence is one batch)
    - Random: batch_size=N (multiple random frames per batch)
    """
    
    if sampling_mode == DatasetSamplingMode.STREAM:
        return dict(
            dataset=dataset,
            batch_size=None,  # ← No batching! One worker processes one seq
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_streaming,
        )
    
    elif sampling_mode == DatasetSamplingMode.RANDOM:
        sampler = get_weighted_random_sampler(dataset) \
            if dataset_config.train.random.weighted_sampling else None
        return dict(
            dataset=dataset,
            batch_size=batch_size,  # ← Batch multiple random frames
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # Drop incomplete batches for consistent size
            collate_fn=custom_collate_rnd,
        )

class DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        sampling_mode = self.dataset_config.train.sampling
        
        if sampling_mode == DatasetSamplingMode.MIXED:
            # Build both loaders
            stream_dataset = build_streaming_dataset(...)
            random_dataset = build_random_access_dataset(...)
            
            stream_loader = DataLoader(
                stream_dataset,
                **get_dataloader_kwargs(..., sampling_mode=STREAM)
            )
            
            random_loader = DataLoader(
                random_dataset,
                **get_dataloader_kwargs(..., sampling_mode=RANDOM)
            )
            
            # Mix them together
            from torch.utils.data import ChainDataset
            
            # Simple approach: alternate batches
            class MixedLoader:
                def __init__(self, stream_loader, random_loader,
                            w_stream, w_random):
                    self.stream_iter = iter(stream_loader)
                    self.random_iter = iter(random_loader)
                    self.w_stream = w_stream
                    self.w_random = w_random
                    self.total_weight = w_stream + w_random
                
                def __iter__(self):
                    return self
                
                def __next__(self):
                    if random() < self.w_stream / self.total_weight:
                        return next(self.stream_iter)
                    else:
                        return next(self.random_iter)
            
            return MixedLoader(stream_loader, random_loader,
                              self.config.train.mixed.w_stream,
                              self.config.train.mixed.w_random)
        
        elif sampling_mode == DatasetSamplingMode.RANDOM:
            random_dataset = build_random_access_dataset(...)
            return DataLoader(
                random_dataset,
                **get_dataloader_kwargs(..., sampling_mode=RANDOM)
            )
        
        elif sampling_mode == DatasetSamplingMode.STREAM:
            stream_dataset = build_streaming_dataset(...)
            return DataLoader(
                stream_dataset,
                **get_dataloader_kwargs(..., sampling_mode=STREAM)
            )
```

### 3.2 Batch Structure Differences

```python
# STREAMING BATCH (from streaming dataloader)
batch = {
    WORKER_ID_KEY: 0,  # Worker ID
    DATA_KEY: {
        EV_REPR: [
            # L tensors, each [B, C, H, W]
            torch.randn(1, 2, 240, 304),  # Frame 0
            torch.randn(1, 2, 240, 304),  # Frame 1
            ...
            torch.randn(1, 2, 240, 304),  # Frame 9
        ],
        OBJLABELS_SEQ: [
            # L label objects
            SparselyBatchedObjectLabels([obj_label1, None]),  # Frame 0
            SparselyBatchedObjectLabels([obj_label2, obj_label3]),  # Frame 1
            ...
        ],
        IS_FIRST_SAMPLE: torch.tensor([True, False]),  # [B]
        # True only at sequence start
        IS_LAST_SAMPLE: False,
        IS_PADDED_MASK: [
            [False, False],  # Frame 0, both samples valid
            [False, False],  # Frame 1, both samples valid
            ...
        ]
    }
}

# RANDOM BATCH (from random-access dataloader)
batch = {
    WORKER_ID_KEY: 1,
    DATA_KEY: {
        EV_REPR: [
            # 1 "frame" containing B random event reprs
            torch.randn(8, 2, 240, 304),  # B=8 random frames
        ],
        OBJLABELS_SEQ: [
            # 1 label object containing B labels
            SparselyBatchedObjectLabels([label1, label2, ..., label8]),
        ],
        IS_FIRST_SAMPLE: torch.tensor([True, True, True, True, True, True, True, True]),
        # All True because no temporal context
    }
}
```

**Mixed Sampling Benefits:**

```
Training Loss Curve Comparison:

Pure Streaming:
  - Slower convergence (RNN needs time to learn)
  - Good temporal understanding
  - Lower variance at convergence
  Loss: ──────────────────────────▼

Pure Random:
  - Fast convergence (no RNN overhead)
  - Poor temporal understanding
  - Higher variance (overfits)
  Loss: ─────▼──────────────────────

Mixed (50% stream, 50% random):
  - Medium convergence speed
  - Both temporal AND efficient learning
  - Lower variance (better generalization)
  Loss: ───▼─────────────────────────

Result: Mixed sampling achieves best final performance!
```

---

## Part 4: Self-Training Loop Configuration

### 4.1 Dataset Configuration for Weak/Semi-Supervised

**File:** `config/dataset/gen1x0.01_ss.yaml` (WSOD)

```yaml
name: gen1
path: ./datasets/gen1/

# Weakly-supervised: sparse labels on ALL sequences
ratio: 0.01  # Keep 1% of frames labeled

# How to subsample:
# Option 1: Deterministic subsampling per sequence
#   For each sequence with N frames, keep N*0.01 frames
#   Ensures reproducibility across runs
seed: 42  # Reproducibility

train:
  sampling: 'mixed'  # Mix random + streaming
  random:
    weighted_sampling: False  # Each sample equally likely
  mixed:
    w_stream: 1  # 50% streaming batches
    w_random: 1  # 50% random batches

eval:
  sampling: 'stream'  # Always streaming for metrics
```

**File:** `config/dataset/gen1x0.01_ss-1round.yaml` (After pseudo-labeling)

```yaml
name: gen1
# After first self-training round
path: ./datasets/pseudo_gen1/gen1x0.01_ss-1round/train/
# This path has: event files (soft-linked) + pseudo-label .npy files

# NEW: Use all pseudo-labels (ratio=-1)
ratio: -1  # Use all labels (no subsampling)

train:
  sampling: 'mixed'  # Still mix for training

eval:
  sampling: 'stream'
```

### 4.2 Model Configuration for Hard vs Soft Targets

**File:** `config/model/rnndet.yaml` (WSOD - Hard Targets)

```yaml
model:
  name: rnndet
  
  head:
    num_classes: 2
    soft_targets: False  # ← Hard anchor assignment
    anchor_size: [10, 20, 40]  # Anchor scales
  
  # Loss weights
  loss_weights:
    objectness: 1.0  # Binary classification (object or not)
    class: 1.0  # Multi-class classification
    bbox: 5.0  # Regression
```

**File:** `config/model/rnndet-soft.yaml` (SSOD - Soft Targets)

```yaml
model:
  name: rnndet-soft
  
  head:
    num_classes: 2
    soft_targets: True  # ← Soft anchor assignment
    soft_label_weight: 0.5  # How much to trust pseudo-labels
    anchor_size: [10, 20, 40]
  
  loss_weights:
    objectness: 1.0
    class: 1.0
    bbox: 5.0
```

**What's the Difference?**

```python
# WSOD (Hard Targets)
# Ground truth bbox at position (100, 100, 50, 50)
# Model predicts: anchor @ (99, 101, 51, 49) with conf=0.6

loss = (conf - 1.0)^2  # ← Force confidence to 1.0
     + (bbox - gt_bbox)^2  # ← Force bbox to exact GT

# Problem with pseudo-labels:
# Pseudo label bbox might be slightly wrong
# Hard targets would force model to learn the error!

# SSOD (Soft Targets)
# Pseudo-label bbox: (98, 102, 52, 48) with pred_conf=0.6
# Model predicts: anchor @ (99, 101, 51, 49) with conf=0.65

loss = (conf - 0.6)^2  # ← Only target the predicted confidence
     + (bbox - pseudo_bbox)^2  # ← Allow some flexibility

# Better! Model doesn't get punished for pseudo-label errors
# As long as it's close to the pseudo-label's estimate
```

### 4.3 Training Hyperparameters by Round

```yaml
# Round 0: Supervised Baseline (small amount of GT labels)
config/experiment/gen1/default.yaml:
  training:
    max_steps: 200000  # 200k steps for 1% data
    learning_rate: 0.0002  # Standard LR
    lr_scheduler:
      pct_start: 0.005  # 5% warmup
      div_factor: 20  # LR ramp-up

# Round 1: SSOD Training (pseudo-labels only)
config/experiment/gen1/default.yaml (with overrides):
  training:
    max_steps: 150000  # Fewer steps (denser labels)
    learning_rate: 0.0005  # Higher LR (faster convergence)
    # Pseudo-labels are denser, so converges faster
```

**Step Configuration by Data Ratio:**

```python
# From paper Table in Appendix A.2
steps_by_ratio = {
    0.01: 200000,  # 1% data
    0.02: 300000,  # 2% data
    0.05: 400000,  # 5% data
    0.10: 400000,  # 10% data
    1.00: 400000,  # Full data (original RVT)
}

# Reasoning:
# - More labeled data → model learns faster → fewer steps needed
# - But beyond certain point, diminishing returns
# - 5% and above: use same 400k steps as full training
```

---

## Part 5: Quick Reference - Key Functions and Their Purposes

| Function | Location | Purpose |
|----------|----------|---------|
| `RNNDetector.forward()` | `models/detection/recurrent_backbone/maxvit_rnn.py:97-115` | Forward pass through 4 recurrent stages |
| `Module.training_step()` | `modules/detection.py:150-298` | One training iteration on batch |
| `PseudoLabelModule.validation_step()` | `modules/pseudo_labeler.py:325-380` | Generate pseudo-labels for one sequence |
| `tta_postprocess()` | `modules/pseudo_labeler.py:37-91` | Merge TTA predictions via NMS |
| `EventSeqData._track()` | `modules/pseudo_labeler.py:201-260` | Filter pseudo-labels via tracking |
| `LinearBoxTracker.update()` | `modules/tracking/linear.py:80-99` | Update tracker with detection |
| `DataModule.train_dataloader()` | `modules/data/genx.py:160-180` | Create mixed data loader |
| `get_subsample_label_idx()` | `modules/utils/ssod.py:19-37` | Generate label subsampling indices |
| `PropheseeEvaluator.evaluate_buffer()` | `utils/evaluation/prophesee/evaluator.py` | Compute COCO AP metrics |

---

## Debugging Tips

### 1. RNN State Issues

**Problem:** Gradient explosion during training
**Solution:** Check `save_states_and_detach()` is detaching properly

```python
# Verify states are detached
assert not prev_states[0][0].requires_grad  # h should not require grad
assert not prev_states[0][1].requires_grad  # c should not require grad
```

### 2. Pseudo-Label Quality

**Problem:** Model performance decreases after pseudo-labeling
**Cause:** Pseudo-labels are too noisy
**Solution:** 
- Increase `confidence_threshold` in filter config
- Increase `min_track_len` in tracking
- Evaluate pseudo-label precision before re-training

```bash
python val_dst.py model=pseudo_labeler dataset=gen1x0.01_ss-1round \
  model.pseudo_label.obj_thresh=0.05  # Higher threshold
  model.pseudo_label.cls_thresh=0.05
```

### 3. Memory Issues During Mixed Sampling

**Problem:** OOM error during training
**Cause:** Streaming batch consumes more memory than random batch
**Solution:** Reduce `w_stream` weight or reduce `sequence_length`

```yaml
dataset:
  train:
    mixed:
      w_stream: 0.5  # 33% streaming
      w_random: 1.0  # 67% random
  
  sequence_length: 5  # Process 5 frames at a time instead of 10
```

---

## Conclusion

The LEOD implementation demonstrates production-quality deep learning code with:

1. **Proper state management** - RNN states with clear semantics
2. **Robust pseudo-label generation** - Multi-stage filtering pipeline
3. **Flexible data loading** - Mixed sampling for training efficiency
4. **Reproducible experiments** - Hydra config management
5. **Standard evaluation** - Prophesee/COCO metrics integration

Understanding these key modules is essential for:
- Extending the method to new domains
- Debugging training issues
- Reproducing paper results
- Contributing improvements

