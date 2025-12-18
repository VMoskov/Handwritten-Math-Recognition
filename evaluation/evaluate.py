import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


class ObjectDetectionEvaluator:
    def __init__(self, class_names, device):
        '''
        Args:
            class_names (list): List of class names
                                Index 0 should be background.
            device (torch.device): CPU or Cuda.
        '''
        self.device = device
        self.class_names = class_names
        
        self.map_metric = MeanAveragePrecision(class_metrics=True).to(device)
        
        self.iou_sums = {i: 0.0 for i in range(len(class_names))}
        self.gt_counts = {i: 0 for i in range(len(class_names))}

    def update(self, predictions, targets):
        '''
        Args:
            predictions (list): List of dicts from model output.
            targets (list): List of dicts from dataloader.
        '''
        preds_formatted = [{k: v.to(self.device) for k, v in p.items()} for p in predictions]
        targets_formatted = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        self.map_metric.update(preds_formatted, targets_formatted)

        for pred, target in zip(preds_formatted, targets_formatted):
            self._update_iou_stats(pred, target)

    def _update_iou_stats(self, pred, target):
        '''
        Calculates the best IoU for each Ground Truth box.
        '''
        p_boxes = pred['boxes']
        p_labels = pred['labels']
        t_boxes = target['boxes']
        t_labels = target['labels']

        if len(t_boxes) == 0:
            return

        # no predictions -> IoU = 0
        if len(p_boxes) == 0:
            for label in t_labels:
                lbl = label.item()
                self.gt_counts[lbl] += 1
            return

        ious = box_iou(p_boxes, t_boxes)

        # find best IoU for each GT box
        for t_idx, t_label in enumerate(t_labels):
            lbl = t_label.item()
            self.gt_counts[lbl] += 1
            
            same_class_indices = (p_labels == t_label).nonzero(as_tuple=True)[0]
            
            if len(same_class_indices) > 0:
                class_ious = ious[same_class_indices, t_idx]
                best_iou = class_ious.max().item()
                self.iou_sums[lbl] += best_iou
            else:  
                # no prediction of the correct class -> IoU = 0
                self.iou_sums[lbl] += 0.0

    def compute(self):
        map_results = self.map_metric.compute()
        
        iou_results = {}
        total_iou_sum = 0
        total_gt_count = 0
        
        for i, name in enumerate(self.class_names):
            if i == 0: continue # skip background
            
            count = self.gt_counts[i]
            if count > 0:
                avg_iou = self.iou_sums[i] / count
                iou_results[f'IoU/{name}'] = avg_iou
                
                total_iou_sum += self.iou_sums[i]
                total_gt_count += count
            else:
                iou_results[f'IoU/{name}'] = -1.0  # -1 if no GT instances
        
        if total_gt_count > 0:
            iou_results['mIoU'] = total_iou_sum / total_gt_count
        else:
            iou_results['mIoU'] = 0.0

        # Combine results
        final_results = {
            'mAP': map_results['map'],
            'mAP@75': map_results['map_75'],
            'mAP@50': map_results['map_50'],
            'mIoU': iou_results['mIoU'],
            'per_class_iou': iou_results,
            'per_class_map': map_results['map_per_class']
        }
        
        return final_results

    def reset(self):
        self.map_metric.reset()
        self.iou_sums = {i: 0.0 for i in range(len(self.class_names))}
        self.gt_counts = {i: 0 for i in range(len(self.class_names))}