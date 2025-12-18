import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import os
from pathlib import Path
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.CROHMEDetectionDataset import (
    FullExpressionsDataset, TrainingSymbolsDataset, collate_fn, get_classifier_transforms, get_detector_transforms
)
from data.SyntheticDataset import SyntheticMathDataset

from model.detector import get_detector_model
from model.classifier import get_classifier_model
from model.translator import get_translator_model
from model.pipeline_model import PipelineModel


class Trainer:
    def __init__(self, model, detector_loaders, classifier_loader, translator_loaders, config, detector_criterion=None):
        '''
        Args:
            model: The PipelineModel containing detector and classifier heads
            detector_loaders: Tuple (train_loader, val_loader) for the Detector (Full Equations)
            translator_loaders: Tuple (train_loader, val_loader) for the Translator (Box-to-LaTeX)
            classifier_loaders: Tuple (train_loader, val_loader) for the Classifier (Iso Symbols)
            device: torch.device
            detector_criterion: Loss function for the detector (optional, FasterRCNN has built-in loss)
        '''
        self.config = config
        self.device = self.get_device(config['system'].get('device', 'auto'))
        self.model = model.to(self.device)

        # unpacking data loaders
        self.detector_train_loader, self.detector_val_loader = detector_loaders
        self.classifier_train_loader, self.classifier_val_loader = classifier_loader
        self.translator_train_loader, self.translator_val_loader = translator_loaders

        det_cfg = self.config['training']['detector']
        clf_cfg = self.config['training']['classifier']
        trans_cfg = self.config['training']['translator']

        self.detector_optimizer = optim.SGD(
            self.model.detection_head.parameters(), 
            lr=det_cfg['optimizer']['lr'],
            momentum=det_cfg['optimizer']['momentum'],
            weight_decay=det_cfg['optimizer'].get('weight_decay', 0.0)
        )
        self.classifier_optimizer = optim.Adam(
            self.model.classification_head.parameters(), 
            lr=clf_cfg['optimizer']['lr'],
            weight_decay=clf_cfg['optimizer'].get('weight_decay', 0.0)
        )
        self.translator_optimizer = optim.Adam(
            self.model.translation_head.parameters(),
            lr=trans_cfg['optimizer']['lr'],
            weight_decay=trans_cfg.get('weight_decay', 0.0)
        )

        self.detector_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.detector_optimizer,
            mode=det_cfg['scheduler']['mode'],
            factor=det_cfg['scheduler']['factor'],
            patience=det_cfg['scheduler']['patience']
        )
        self.classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.classifier_optimizer,
            mode=clf_cfg['scheduler']['mode'],
            factor=clf_cfg['scheduler']['factor'],
            patience=clf_cfg['scheduler']['patience']
        )
        self.translator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.translator_optimizer,
            mode=trans_cfg['scheduler']['mode'],
            factor=trans_cfg['scheduler']['factor'],
            patience=trans_cfg['scheduler']['patience']
        )

        self.detector_criterion = detector_criterion
        self.classifier_criterion = nn.CrossEntropyLoss()
        self.translator_criterion = nn.CrossEntropyLoss(ignore_index=self.model.PAD_IDX)

        self.save_dir = Path(config['system']['save_dir'])
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_config(path='config.yaml'):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
        
    @staticmethod
    def get_device(device_str='auto'):
        if device_str == 'auto':
            if torch.cuda.is_available(): return torch.device('cuda')
            if torch.backends.mps.is_available(): return torch.device('mps')
            return torch.device('cpu')
        return torch.device(device_str)
    
    @staticmethod
    def build_data_loaders(config):
        '''
        Creates all Datasets and DataLoaders based on paths in config.yaml
        Returns: 
            det_loaders: (train_loader, val_loader)
            clf_loaders: (train_loader, val_loader)
            det_class_names: List[str]
            clf_class_names: List[str]
        '''
        print('Building DataLoaders from config...')
        data_cfg = config['data']
        
        # detection data
        det_transform = get_detector_transforms()

        det_dataset = FullExpressionsDataset(
            root_dir=data_cfg['detection_root'], 
            transform=det_transform
        )
        
        # split 90/10
        train_size = int(0.9 * len(det_dataset))
        val_size = len(det_dataset) - train_size
        det_train, det_val = torch.utils.data.random_split(det_dataset, [train_size, val_size])
        
        det_loaders = (
            DataLoader(det_train, batch_size=data_cfg['batch_size_detector'], shuffle=True, collate_fn=collate_fn),
            DataLoader(det_val, batch_size=data_cfg['batch_size_detector'], shuffle=False, collate_fn=collate_fn)
        )

        # classification data
        clf_transform = get_classifier_transforms()
        
        clf_dataset = TrainingSymbolsDataset(
            root_dir=data_cfg['classification_root'], 
            transform=clf_transform
        )
        
        # split 80/20
        train_size = int(0.8 * len(clf_dataset))
        val_size = len(clf_dataset) - train_size
        clf_train, clf_val = torch.utils.data.random_split(clf_dataset, [train_size, val_size])
        
        clf_loaders = (
            DataLoader(clf_train, batch_size=data_cfg['batch_size_classifier'], shuffle=True),
            DataLoader(clf_val, batch_size=data_cfg['batch_size_classifier'], shuffle=False)
        )

        vocab_list = clf_dataset.classes

        trans_train_ds = SyntheticMathDataset(vocab_list=vocab_list, num_samples=int(data_cfg['synthetic_samples'] * 0.9))
        trans_val_ds = SyntheticMathDataset(vocab_list=vocab_list, num_samples=int(data_cfg['synthetic_samples'] * 0.1))
        
        trans_loaders = (
            DataLoader(trans_train_ds, batch_size=data_cfg['batch_size_translator'], shuffle=True),
            DataLoader(trans_val_ds, batch_size=data_cfg['batch_size_translator'], shuffle=False)
        )
        
        return det_loaders, clf_loaders, trans_loaders, det_dataset.meta_classes, clf_dataset.classes
    
    @staticmethod
    def build_model(config, det_class_names=None, clf_class_names=None):
        '''
        Factory method to build the Unified Model using the config.
        '''
        print('Building models from config...')
        
        num_det_classes = len(det_class_names) if det_class_names else config['data']['num_meta_classes']
        num_clf_classes = len(clf_class_names) if clf_class_names else config['data']['num_symbol_classes']

        detector = get_detector_model(num_classes=num_det_classes)
        classifier = get_classifier_model(num_classes=num_clf_classes)
        translator = get_translator_model(num_classes=num_clf_classes + 4)  # +4 for special tokens (frac_bar, <sos>, <eos>, <PAD>)

        if config['model']['detector_weights']:
            print(f'Loading detector weights: {config['model']['detector_weights']}')
            detector.load_state_dict(torch.load(config['model']['detector_weights'], weights_only=True))
        else:
            print('No detector weights provided, training from scratch.')
            
        if config['model']['classifier_weights']:
            print(f'Loading classifier weights: {config['model']['classifier_weights']}')
            classifier.load_state_dict(torch.load(config['model']['classifier_weights'], weights_only=True))
        else:
            print('No classifier weights provided, training from scratch.')

        if config['model']['translator_weights']:
            print(f'Loading translator weights: {config['model']['translator_weights']}')
            translator.load_state_dict(torch.load(config['model']['translator_weights'], weights_only=True))

        model = PipelineModel(
            detection_head=detector,
            classification_head=classifier,
            translation_head=translator,
            meta_classes=det_class_names,
            symbol_classes=clf_class_names
        )
        
        return model

    def _train_detector_epoch(self):
        self.model.detection_head.train()
        total_loss = 0

        loop = tqdm(self.detector_train_loader, desc='Detector Train', leave=False)
        
        for images, targets in loop:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.detector_optimizer.zero_grad()
            
            outputs = self.model.detection_head(images, targets)
            
            # model computes its own loss (FasterRCNN style)
            if isinstance(outputs, dict) and 'loss_classifier' in outputs:
                loss = sum(loss for loss in outputs.values())
            
            # model returns predictions only, compute loss externally
            else:
                if self.det_criterion is None:
                    raise ValueError("Model returned predictions but no 'detector_criterion' was provided!")
                loss = self.det_criterion(outputs, targets)
            
            loss.backward()
            self.detector_optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.detector_train_loader)
    
    def _validate_detector_epoch(self):
        self.model.detection_head.train()  # keep in train mode to get loss
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in self.detector_val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model.detection_head(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                total_loss += loss.item()
                
        return total_loss / len(self.detector_val_loader)
    
    def _train_classifier_epoch(self):
        self.model.classification_head.train()
        total_loss = 0

        loop = tqdm(self.classifier_train_loader, desc='Classifier Train', leave=False)
        
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)

            self.classifier_optimizer.zero_grad()
            outputs = self.model.classification_head(images)
            loss = self.classifier_criterion(outputs, labels)
            loss.backward()
            self.classifier_optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.classifier_train_loader)
    
    def _validate_classifier_epoch(self):
        self.model.classification_head.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.classifier_val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model.classification_head(images)
                loss = self.classifier_criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return total_loss / len(self.classifier_val_loader), 100 * correct / total
    
    def train_translator_epoch(self):
        self.model.translation_head.train()
        total_loss = 0
        
        loop = tqdm(self.translator_train_loader, desc='Translator Train', leave=False)
        
        for batch in loop:
            src_boxes = batch['input_boxes'].to(self.device)
            target_strings = batch['target_latex']
            
            tgt_tokens = self._tokenize_batch(target_strings) # (Batch, Seq_Len)
            tgt_tokens = tgt_tokens.to(self.device)
            
            tgt_input = tgt_tokens[:, :-1]
            tgt_output = tgt_tokens[:, 1:]
            
            self.translator_optimizer.zero_grad()
            
            src_cls = src_boxes[:, :, 0].long()
            src_key_padding_mask = (src_cls == self.model.PAD_IDX)
            
            tgt_key_padding_mask = (tgt_input == self.model.PAD_IDX)
            
            seq_len = tgt_input.size(1)
            tgt_mask = self.model.translation_head.generate_square_subsequent_mask(seq_len).to(self.device)
            
            logits = self.model.translation_head(
                src_boxes=src_boxes,
                tgt_tokens=tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask
            )
            
            loss = self.translator_criterion(
                logits.reshape(-1, logits.shape[-1]), 
                tgt_output.reshape(-1)
            )
            
            loss.backward()
            self.translator_optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.translator_train_loader)
    
    def _validate_translator_epoch(self):
        self.model.translation_head.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.translator_val_loader:
                src_boxes = batch['input_boxes'].to(self.device)
                tgt_tokens = self._tokenize_batch(batch['target_latex']).to(self.device)
                
                tgt_input = tgt_tokens[:, :-1]
                tgt_output = tgt_tokens[:, 1:]

                src_cls = src_boxes[:, :, 0].long()
                src_key_padding_mask = (src_cls == self.model.PAD_IDX)
                tgt_key_padding_mask = (tgt_input == self.model.PAD_IDX)

                seq_len = tgt_input.size(1)
                tgt_mask = self.model.translation_head.generate_square_subsequent_mask(seq_len).to(self.device)
                
                logits = self.model.translation_head(
                    src_boxes=src_boxes,
                    tgt_tokens=tgt_input,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_mask=tgt_mask
                )
                loss = self.translator_criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
                total_loss += loss.item()
        return total_loss / len(self.translator_val_loader)
    
    def _tokenize_batch(self, strings):
        '''
        Converts a list of LaTeX strings into a padded Tensor of IDs.
        Adds START, END, and handles padding.
        '''
        batch_ids = []
        for s in strings:
            s = s.replace('{', ' { ').replace('}', ' } ')
            tokens = s.split()
            
            ids = [self.model.START_IDX]
            for t in tokens:
                if t in self.model.token_to_id:
                    ids.append(self.model.token_to_id[t])
                else:
                    pass
            ids.append(self.model.END_IDX)
            batch_ids.append(torch.tensor(ids))
        
        # Pad batch
        padded = pad_sequence(batch_ids, batch_first=True, padding_value=self.model.PAD_IDX)
        return padded
    
    def train_detector(self):
        cfg = self.config['training']['detector']
        epochs = cfg['epochs']
        patience = cfg['patience']

        save_path = self.save_dir / 'detector'
        save_path.mkdir(exist_ok=True)

        print('Training Detector Head...')
        for param in self.model.detection_head.parameters(): param.requires_grad = True
        for param in self.model.classification_head.parameters(): param.requires_grad = False
        for param in self.model.translation_head.parameters(): param.requires_grad = False
        
        best_det_loss = float('inf')
        best_det_dict = None
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self._train_detector_epoch()
            val_loss = self._validate_detector_epoch()

            self.detector_scheduler.step(val_loss)
            current_lr = self.detector_optimizer.param_groups[0]['lr']
            
            print(f'[Detector] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}', end='')

            if val_loss < best_det_loss:
                best_det_loss = val_loss
                patience_counter = 0  # reset counter
                best_det_dict = self.model.detection_head.state_dict()
                tmp_path = save_path / f'best_detector_epoch_{epoch+1}_{best_det_loss:.02}.pth'
                torch.save(best_det_dict, tmp_path)
                print(' -> Saved Best!')
            else:
                patience_counter += 1
                print(f' | Patience {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                print('>> Early Stopping Triggered for Detector.')
                break
        
        save_path = save_path / f'best_detector_{best_det_loss:.02}.pth'

        print(f'\nSaving Best Detector Weights to {save_path} ...')
        torch.save(best_det_dict, save_path)

        print('Reloading best detector weights...')
        self.model.detection_head.load_state_dict(best_det_dict)

        print('\nDetector Training Complete.')

    def train_classifier(self):
        cfg = self.config['training']['classifier']
        epochs = cfg['epochs']
        patience = cfg['patience']

        save_path = self.save_dir / 'classifier'
        save_path.mkdir(exist_ok=True)

        print('Training Classifier Head...')
        for param in self.model.detection_head.parameters(): param.requires_grad = False
        for param in self.model.classification_head.parameters(): param.requires_grad = True
        for param in self.model.translation_head.parameters(): param.requires_grad = False

        best_clf_acc = 0.0
        best_clf_dict = None
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self._train_classifier_epoch()
            val_loss, val_acc = self._validate_classifier_epoch()

            self.classifier_scheduler.step(val_loss)
            current_lr = self.classifier_optimizer.param_groups[0]['lr']

            print(f'[Classifier] Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}', end='')

            if val_acc > best_clf_acc:
                best_clf_acc = val_acc
                patience_counter = 0  # reset counter
                best_clf_dict = self.model.classification_head.state_dict()
                tmp_path = save_path / f'best_classifier_epoch_{epoch+1}_{best_clf_acc:.02}.pth'
                torch.save(best_clf_dict, tmp_path)
                print(' -> Saved Best!')
            else:
                patience_counter += 1
                print(f' | Patience {patience_counter}/{patience}')

            if patience_counter >= patience:
                print('>> Early Stopping Triggered for Classifier.')
                break

        save_path = save_path / f'best_classifier_{best_clf_acc:.02}.pth'

        print(f'\nSaving Best Classifier Weights to {save_path} ...')
        torch.save(best_clf_dict, save_path)

        print('Reloading best classifier weights...')
        self.model.classification_head.load_state_dict(best_clf_dict)

        print('\nClassifier Training Complete.')

    def train_translator(self):
        cfg = self.config['training']['translator']
        epochs = cfg['epochs']
        patience = cfg['patience']

        save_path = self.save_dir / 'translator'
        save_path.mkdir(exist_ok=True)
        
        print('\nTraining Translator Head...')
        # Freeze other heads to save memory/compute
        for p in self.model.detection_head.parameters(): p.requires_grad = False
        for p in self.model.classification_head.parameters(): p.requires_grad = False
        for p in self.model.translation_head.parameters(): p.requires_grad = True
        
        best_loss = float('inf')
        best_trans_dict = None
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_translator_epoch()
            val_loss = self._validate_translator_epoch()

            self.translator_scheduler.step(val_loss)
            current_lr = self.translator_optimizer.param_groups[0]['lr']
            
            print(f'[Translator] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}', end='')
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0  # reset counter
                best_trans_dict = self.model.translation_head.state_dict()
                tmp_path = save_path / f'best_translator_epoch_{epoch+1}_{best_loss:.02}.pth'
                torch.save(best_trans_dict, tmp_path)
                print(' -> Saved Best!')
            else:
                patience_counter += 1
                print(f' | Patience {patience_counter}/{patience}')

            if patience_counter >= patience:
                print('>> Early Stopping Triggered for Translator.')
                break

        save_path = save_path / f'best_translator_{best_loss:.02}.pth'
        print(f'\nSaving Best Translator Weights to {save_path} ...')
        torch.save(best_trans_dict, save_path)

        print('Reloading best translator weights...')
        self.model.translation_head.load_state_dict(best_trans_dict)

        print('\nTranslator Training Complete.')
    
    def train(self):
        # self.train_detector()
        self.train_classifier()
        self.train_translator()
        print('\nTraining Complete. Best models loaded into memory.')


def parse_args():
    parser = ArgumentParser(description='Train Pipeline Model for Detection and Classification')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = Trainer.load_config(args.config)

    det_loaders, clf_loaders, trans_loaders, det_class_names, clf_class_names = Trainer.build_data_loaders(config)
    model = Trainer.build_model(config, det_class_names, clf_class_names)

    trainer = Trainer(
        model=model,
        detector_loaders=det_loaders,
        classifier_loader=clf_loaders,
        translator_loaders=trans_loaders,
        config=config
    )

    trainer.train()