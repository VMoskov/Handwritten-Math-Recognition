import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.utils as vutils


class PipelineModel(nn.Module):
    def __init__(self, detection_head, classification_head, translation_head, meta_classes, symbol_classes, classifier_input_size=(32, 32)):
        '''
        A pipeline model that first detects symbols in an image and then classifies each detected symbol.
        
        Args:
            detection_head (nn.Module): A trained object detector (e.g., FasterRCNN, YOLO, SSD).
                                  Must return a list of dicts with 'boxes', 'labels', 'scores'.
            classification_head (nn.Module): A trained image classifier (e.g., ResNet, ViT).
            translation_head (nn.Module): A trained translator model (e.g., Transformer).
            meta_classes (list): List of class names for the detector.
            symbol_classes (list): List of class names for the classifier.
            classifier_input_size (tuple): The input size expected by the classifier (H, W).
        '''
        super(PipelineModel, self).__init__()
        self.detection_head = detection_head
        self.classification_head = classification_head
        self.translation_head = translation_head

        self.meta_classes = meta_classes
        self.symbol_classes = symbol_classes
        self.vocab = symbol_classes
        self.classifier_input_size = classifier_input_size

        self.special_tokens = ['frac_bar', '<sos>', '<eos>', '<PAD>']
        self.full_vocab = self.vocab + self.special_tokens
        self.token_to_id = {k: i for i, k in enumerate(self.full_vocab)}
        self.id_to_token = {i: k for i, k in enumerate(self.full_vocab)}
        
        self.START_IDX = self.token_to_id['<sos>']
        self.END_IDX = self.token_to_id['<eos>']
        self.PAD_IDX = self.token_to_id['<PAD>']

        # normalization parameters for the classifier (ImageNet)
        self.register_buffer('clf_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('clf_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, inputs, confidence_threshold=0.5, max_seq_len=50):
        '''
        Forward pass through the pipeline model.
        
        Args:
            inputs (torch.Tensor): Batch of input images (B, C, H, W).
            confidence_threshold (float): Minimum confidence score to consider a detection valid.
        
        Returns:
            detections (list): List of dicts for each image with keys 'boxes', 'meta_labels', 'symbol_labels', 'scores'.
        '''
        detection_outputs = self.detection_head(inputs)
        results = []

        device = inputs[0].device if isinstance(inputs, list) else inputs.device

        for i, output in enumerate(detection_outputs):
            img = inputs[i]

            masks = output['scores'] >= confidence_threshold
            boxes = output['boxes'][masks]
            meta_labels = output['labels'][masks]
            scores = output['scores'][masks]

            if len(boxes) == 0:
                self._add_empty_result(results)
                continue

            crops, valid_indices = self._crop_and_preprocess(img, boxes)

            if len(crops) == 0:
                self._add_empty_result(results)
                continue

            batch_crops = torch.cat(crops, dim=0)  # (N, C, H, W)

            cls_logits = self.classification_head(batch_crops)
            _, symbol_labels = torch.max(cls_logits, dim=1)

            img_h, img_w = img.shape[1], img.shape[2]
            
            # filter boxes/labels to only valid ones before translating
            valid_boxes = boxes[valid_indices]
            
            latex_expression = self._run_translator(
                valid_boxes, 
                symbol_labels, 
                img_w, img_h, 
                max_seq_len, 
                device=device
            )

            self._assemble_result(results, boxes, scores, output['labels'][masks], symbol_labels, valid_indices, latex_expression)

        return results
    
    def _crop_and_preprocess(self, img, boxes):
        '''
        Crop detected boxes, PAD them to be square (preserving aspect ratio), 
        and then resize to classifier input size.
        '''
        crops = []
        valid_indices = []
        h_img, w_img = img.shape[1], img.shape[2]
        
        # margin to prevent cutting off edges
        margin = 8 
        
        for b_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int().tolist()
            
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w_img, x2 + margin)
            y2 = min(h_img, y2 + margin)
            
            # skip invalid boxes
            if x2 - x1 < 2 or y2 - y1 < 2: 
                continue
                
            crop = img[:, y1:y2, x1:x2]  # shape: (C, H, W)
            
            c, h, w = crop.shape
            max_dim = max(h, w)
            
            pad_h = (max_dim - h) // 2
            pad_w = (max_dim - w) // 2
            
            bg_val = crop.max() 
            
            padding = (pad_w, max_dim - w - pad_w, pad_h, max_dim - h - pad_h)
            
            crop_square = F.pad(crop, padding, value=bg_val)
            
            crop_resized = F.interpolate(
                crop_square.unsqueeze(0), 
                size=self.classifier_input_size,
                mode='bilinear', 
                align_corners=False
            )

            # contrast normalization
            min_val = crop_resized.min().item()
            max_val = crop_resized.max().item()
            if max_val - min_val > 1e-5:
                crop_resized = (crop_resized - min_val) / (max_val - min_val)

            # gamma correction to amplify strokes
            gamma = 0.5
            crop_resized = torch.pow(crop_resized, gamma)
            
            vutils.save_image(crop_resized, f"debug/debug_crop_{b_idx}.png", normalize=True)

            # normalize
            crop_norm = (crop_resized - self.clf_mean) / self.clf_std
            crops.append(crop_norm)
            valid_indices.append(b_idx)
            
        return crops, valid_indices
    
    def _run_translator(self, boxes, labels, img_w, img_h, max_seq_len, device):
        '''
        Helper to run the greedy decode loop for one image's detections.
        '''
        translator_input = self._prepare_translator_input(boxes, labels, img_w, img_h)
        translator_input = translator_input.unsqueeze(0).to(device) # Add batch dim
        
        curr_tokens = torch.tensor([[self.START_IDX]], device=device)
        generated_str = []
        
        with torch.no_grad():
            for _ in range(max_seq_len):
                logits = self.translation_head(translator_input, curr_tokens)
                
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                if next_token_id == self.END_IDX:
                    break
                    
                curr_tokens = torch.cat([curr_tokens, torch.tensor([[next_token_id]], device=device)], dim=1)
                
                token_str = self.id_to_token.get(next_token_id, "")
                
                if token_str not in self.special_tokens:
                    generated_str.append(token_str)
        
        return " ".join(generated_str)

    def _prepare_translator_input(self, boxes, labels, img_w, img_h):
        '''
        Converts Pixels -> Normalized [Label, Cx, Cy, W, H]
        '''
        vectors = []
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.float()
            
            w = x2 - x1
            h = y2 - y1
            cx = x1 + (w / 2)
            cy = y1 + (h / 2)
            
            norm_cx = cx / img_w
            norm_cy = cy / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            # [label, x, y, w, h]
            vectors.append([float(label), norm_cx, norm_cy, norm_w, norm_h])
            
        return torch.tensor(vectors)
    
    def _add_empty_result(self, results):
        results.append({
            'boxes': torch.empty((0, 4)),
            'meta_labels': torch.empty((0,), dtype=torch.long),
            'symbol_labels': torch.empty((0,), dtype=torch.long),
            'scores': torch.empty((0,)),
            'latex': ''
        })

    def _assemble_result(self, results, boxes, scores, meta_labels, symbol_labels, valid_indices, latex_expression):
        filtered_boxes = boxes[valid_indices]
        filtered_scores = scores[valid_indices]
        filtered_meta_labels = meta_labels[valid_indices]
        filtered_symbol_labels = symbol_labels

        results.append({
            'boxes': filtered_boxes,
            'meta_labels': filtered_meta_labels,
            'symbol_labels': filtered_symbol_labels,
            'scores': filtered_scores,
            'latex': latex_expression
        })