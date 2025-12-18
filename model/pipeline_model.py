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
        
        # Margin to add around the box (prevents cutting off edges)
        margin = 4 
        
        for b_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int().tolist()
            
            # 1. Add Margin & Clip to Image Bounds
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w_img, x2 + margin)
            y2 = min(h_img, y2 + margin)
            
            # Skip invalid boxes
            if x2 - x1 < 2 or y2 - y1 < 2: 
                continue
                
            crop = img[:, y1:y2, x1:x2] # Shape: (C, H, W)
            
            # 2. Pad to Square (Preserve Aspect Ratio)
            c, h, w = crop.shape
            max_dim = max(h, w)
            
            pad_h = (max_dim - h) // 2
            pad_w = (max_dim - w) // 2
            
            # Determine background color automatically (median of corners) or assume 1.0/0.0
            # Heuristic: The image is likely background-dominant. Max value usually works for white background.
            bg_val = crop.max() 
            
            # Pad: (left, right, top, bottom)
            # We add the extra pixel to the second side if odd padding is needed
            padding = (pad_w, max_dim - w - pad_w, pad_h, max_dim - h - pad_h)
            
            crop_square = F.pad(crop, padding, value=bg_val)
            
            # 3. Resize to Target Size (e.g. 32x32)
            # Now that it is square, resizing won't distort the shape
            crop_resized = F.interpolate(
                crop_square.unsqueeze(0), 
                size=self.classifier_input_size,
                mode='bilinear', 
                align_corners=False
            )
            
            vutils.save_image(crop_resized, f"debug/debug_crop_{b_idx}.png", normalize=True)

            # Normalize
            crop_norm = (crop_resized - self.clf_mean) / self.clf_std
            crops.append(crop_norm)
            valid_indices.append(b_idx)
            
        return crops, valid_indices
    
    def _run_translator(self, boxes, labels, img_w, img_h, max_seq_len, device):
        '''
        Helper to run the greedy decode loop for one image's detections.
        '''
        # A. Prepare Input Vector (Batch=1, N_Boxes, 5)
        translator_input = self._prepare_translator_input(boxes, labels, img_w, img_h)
        translator_input = translator_input.unsqueeze(0).to(device) # Add batch dim
        
        # B. Greedy Decode Loop
        curr_tokens = torch.tensor([[self.START_IDX]], device=device)
        generated_str = []
        
        # We disable gradients for the loop since this is pure inference
        with torch.no_grad():
            for _ in range(max_seq_len):
                # Forward pass
                logits = self.translation_head(translator_input, curr_tokens)
                
                # Get prediction for last token
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                # Stop if END token
                if next_token_id == self.END_IDX:
                    break
                    
                # Append to sequence
                curr_tokens = torch.cat([curr_tokens, torch.tensor([[next_token_id]], device=device)], dim=1)
                
                # Decode ID -> String
                token_str = self.id_to_token.get(next_token_id, "")
                
                # Filter out special tokens from final string
                if token_str not in self.special_tokens:
                    generated_str.append(token_str)
        
        # Return space-separated string
        return " ".join(generated_str)

    def _prepare_translator_input(self, boxes, labels, img_w, img_h):
        '''
        Converts Pixels -> Normalized [Label, Cx, Cy, W, H]
        '''
        vectors = []
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.float()
            
            # Calculate Center + Width/Height
            w = x2 - x1
            h = y2 - y1
            cx = x1 + (w / 2)
            cy = y1 + (h / 2)
            
            # Normalize (0.0 - 1.0)
            norm_cx = cx / img_w
            norm_cy = cy / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            # [Label, x, y, w, h]
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
