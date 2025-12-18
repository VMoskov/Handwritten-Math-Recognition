import torch
from tqdm import tqdm
from train import Trainer
from argparse import ArgumentParser
from evaluation.evaluator import ObjectDetectionEvaluator


def evaluate_system(config_path):
    config = Trainer.load_config(config_path)
    device = Trainer.get_device(config['system']['device'])
    
    det_loaders, _, _, meta_classes, symbol_classes = Trainer.build_data_loaders(config)
    _, val_loader = det_loaders
    
    model = Trainer.build_model(config, det_class_names=meta_classes, clf_class_names=symbol_classes)
    
    model.to(device)
    model.eval()

    meta_evaluator = ObjectDetectionEvaluator(meta_classes, device)

    raw_dataset = val_loader.dataset
    if hasattr(raw_dataset, 'dataset'):  # unwrap if wrapped in Subset
        raw_dataset = raw_dataset.dataset
    dataset_class_names = raw_dataset.fine_grained_classes
    id_translator = {}
    
    print("Building ID Translator...")
    for ds_id, name in enumerate(dataset_class_names):
        if name == '__background__': continue
        
        if name in symbol_classes:
            model_id = symbol_classes.index(name)
            id_translator[ds_id] = model_id
        else:
            id_translator[ds_id] = -1  # unknown class
    
    symbol_classes = ['__background__'] + symbol_classes 
    pipeline_evaluator = ObjectDetectionEvaluator(symbol_classes, device)

    print('Running Evaluation...')

    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = [img.to(device) for img in images]
            
            det_outputs = model.detection_head(images)
            
            preds_meta = []
            preds_pipeline = []
            targets_meta = []
            targets_pipeline = []

            for i, output in enumerate(det_outputs):
                # filter detections
                keep = output['scores'] > 0.5
                boxes = output['boxes'][keep]
                scores = output['scores'][keep]
                meta_labels = output['labels'][keep]

                preds_meta.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': meta_labels
                })

                # pipeline predictions
                if len(boxes) > 0:
                    crops, valid_indices = model._crop_and_preprocess(images[i], boxes)
                    
                    if len(crops) > 0:
                        batch_crops = torch.cat(crops, dim=0)
                        cls_logits = model.classification_head(batch_crops)
                        cls_probs = torch.softmax(cls_logits, dim=1)
                        max_probs, sym_labels = torch.max(cls_probs, dim=1)
                        
                        # adjust labels (shift by +1 to account for background class)
                        sym_labels = sym_labels + 1 
                        
                        final_boxes = boxes[valid_indices]
                        final_scores = scores[valid_indices] * max_probs

                        preds_pipeline.append({
                            'boxes': final_boxes,
                            'scores': final_scores,
                            'labels': sym_labels
                        })
                    else:
                        preds_pipeline.append(dict(boxes=torch.tensor([]).to(device), scores=torch.tensor([]).to(device), labels=torch.tensor([]).to(device)))
                else:
                    preds_pipeline.append(dict(boxes=torch.tensor([]).to(device), scores=torch.tensor([]).to(device), labels=torch.tensor([]).to(device)))

                t_boxes = targets[i]['boxes'].to(device)
                t_meta = targets[i]['labels'].to(device)
                targets_meta.append({'boxes': t_boxes, 'labels': t_meta})

                # pipeline targets
                if 'specific_labels' in targets[i]:
                    raw_ids = targets[i]['specific_labels'].cpu().numpy()
                    translated_ids = []
                    valid_box_indices = []

                    for idx, rid in enumerate(raw_ids):
                        model_id = id_translator.get(rid, -1)
                        if model_id != -1:
                            translated_ids.append(model_id + 1)
                            valid_box_indices.append(idx)

                    if len(translated_ids) > 0:
                        t_sym = torch.tensor(translated_ids).to(device)
                        t_boxes = t_boxes[valid_box_indices]
                        targets_pipeline.append({'boxes': t_boxes, 'labels': t_sym})
                    else:
                        targets_pipeline.append({'boxes': torch.tensor([]).to(device), 'labels': torch.tensor([]).to(device)})

                else:  # fallback
                    targets_pipeline.append({'boxes': t_boxes, 'labels': torch.zeros_like(t_meta)})

            meta_evaluator.update(preds_meta, targets_meta)
            pipeline_evaluator.update(preds_pipeline, targets_pipeline)

    res_meta = meta_evaluator.compute()
    res_pipe = pipeline_evaluator.compute()

    print('\n' + '='*50)
    print('      [1] DETECTOR (META-CLASS) EVALUATION')
    print('='*50)
    print(f'mAP@50:   {res_meta['mAP@50']:.4f}')
    print(f'mIoU:     {res_meta['mIoU']:.4f}')
    print(f'Accuracy: {res_meta['Accuracy']:.2%} (Digits/Letters/Ops)')

    print('\n' + '='*50)
    print('      [2] PIPELINE (SYMBOL) EVALUATION')
    print('='*50)
    print(f'mAP@50:   {res_pipe['mAP@50']}')
    print(f'mIoU:     {res_pipe['mIoU']}')
    print(f'Accuracy: {res_pipe['Accuracy']} (Classifier Accuracy on Detected Boxes)')
    print('='*50)


def parse_args():
    parser = ArgumentParser(description='Evaluate Pipeline Model for Detection and Classification')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evaluate_system(config_path=args.config)