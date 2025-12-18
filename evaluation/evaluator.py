import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


class ObjectDetectionEvaluator:
    def __init__(self, class_names, device):
        '''
        Args:
            class_names (list): List of class names. Index 0 must be background.
            device (torch.device): CPU or Cuda.
        '''
        self.device = device
        self.class_names = class_names
        
        self.map_metric = MeanAveragePrecision(class_metrics=True).to(device)
        
        self.iou_sums = {i: 0.0 for i in range(len(class_names))}
        self.gt_counts = {i: 0 for i in range(len(class_names))}
        
        self.total_detected_boxes = 0
        self.correct_classified_boxes = 0

    def update(self, predictions, targets):
        '''
        Args:
            predictions (list): List of dicts with keys 'boxes', 'labels', 'scores'.
            targets (list): List of dicts with keys 'boxes', 'labels'.
        '''
        preds_formatted = [{k: v.to(self.device) for k, v in p.items()} for p in predictions]
        targets_formatted = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        self.map_metric.update(preds_formatted, targets_formatted)

        for pred, target in zip(preds_formatted, targets_formatted):
            self._update_iou_and_accuracy(pred, target)

    def _update_iou_and_accuracy(self, pred, target):
        '''
        Calculates mIoU (tightness) and Classification Accuracy.
        '''
        p_boxes, p_labels = pred['boxes'], pred['labels']
        t_boxes, t_labels = target['boxes'], target['labels']

        if len(t_boxes) == 0:
            return

        # handle empty predictions
        if len(p_boxes) == 0:
            for label in t_labels:
                self.gt_counts[label.item()] += 1
            return

        ious = box_iou(p_boxes, t_boxes)

        for t_idx, t_label in enumerate(t_labels):
            lbl = t_label.item()
            self.gt_counts[lbl] += 1
            
            same_class_indices = (p_labels == t_label).nonzero(as_tuple=True)[0]
            
            if len(same_class_indices) > 0:
                class_ious = ious[same_class_indices, t_idx]
                best_iou = class_ious.max().item()  # get best IoU for this GT box
                self.iou_sums[lbl] += best_iou

        best_iou_vals, best_iou_indices = ious.max(dim=0)
        
        for t_idx, iou in enumerate(best_iou_vals):
            if iou > 0.5:  # detection threshold
                self.total_detected_boxes += 1
                
                pred_idx = best_iou_indices[t_idx]
                pred_lbl = p_labels[pred_idx]
                gt_lbl = t_labels[t_idx]
                
                if pred_lbl == gt_lbl:
                    self.correct_classified_boxes += 1

    def compute(self):
        map_results = self.map_metric.compute()
        
        iou_results = {}
        total_iou_sum = 0
        total_gt_count = 0
        
        for i, name in enumerate(self.class_names):
            if i == 0: continue  # skip background
            
            count = self.gt_counts[i]
            if count > 0:
                avg_iou = self.iou_sums[i] / count
                iou_results[f'IoU/{name}'] = avg_iou
                total_iou_sum += self.iou_sums[i]
                total_gt_count += count
            else:
                iou_results[f'IoU/{name}'] = -1.0  # no GT found
        
        mean_iou = total_iou_sum / total_gt_count if total_gt_count > 0 else 0.0

        if self.total_detected_boxes > 0:
            clf_acc = self.correct_classified_boxes / self.total_detected_boxes
        else:
            clf_acc = 0.0

        return {
            'mAP': map_results['map'].item(),
            'mAP@50': map_results['map_50'].item(),
            'mIoU': mean_iou,
            'Accuracy': clf_acc,
            'per_class_map': map_results['map_per_class'],
            'per_class_iou': iou_results
        }

    def reset(self):
        self.map_metric.reset()
        self.iou_sums = {i: 0.0 for i in range(len(self.class_names))}
        self.gt_counts = {i: 0 for i in range(len(self.class_names))}
        self.total_detected_boxes = 0
        self.correct_classified_boxes = 0