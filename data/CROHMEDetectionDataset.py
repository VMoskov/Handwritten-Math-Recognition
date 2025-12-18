import torch
import os
import glob
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt


class FullExpressionsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        '''
        Args:
            root_dir (string): Directory with all the images and LG files.
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_files = sorted(glob.glob(os.path.join(root_dir, '*.png')))
        
        self.meta_classes = ['__background__', 'Digit', 'Letter', 'Operator']
        self.meta_class_to_idx = {name: i for i, name in enumerate(self.meta_classes)}

        self.fine_grained_classes = ['__background__']
        self.fine_grained_to_idx = {'__background__': 0}
        
        print('Scanning dataset to build fine-grained class map...')
        self._build_fine_grained_map()
        self._fix_labels()
        print(f'Done. Found {len(self.fine_grained_classes)} unique specific symbols.')
        print(f'Dataset size after filtering: {len(self.image_files)} images.')

    def _build_fine_grained_map(self):
        '''
        Scans all LG files to build a mapping of specific symbol strings to IDs.
        Also filters out images containing invalid labels (with ':').
        '''
        valid_files = []
        
        for img_path in self.image_files:
            lg_path = img_path.replace('.png', '.lg')
            if not os.path.exists(lg_path):
                continue
            
            is_valid_file = True
            
            with open(lg_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    # only process bounding box lines
                    if not row or row[0] != 'BB': continue
                    
                    try:
                        # <label>_<instance_id> -> extract <label>
                        # Changed to rsplit to match parse_lg_file logic and handle underscores correctly
                        label_str = row[1].strip().rsplit('_', 1)[0]

                        # if label_str contains \ remove it
                        label_str = label_str.replace('\\', '')

                        # files containing : or two digit numbers in labels are errors
                        if ':' in label_str or (len(label_str) == 2 and label_str.isdigit()):
                            is_valid_file = False
                            break
                        
                        # add to map if new
                        if label_str not in self.fine_grained_to_idx:
                            self.fine_grained_to_idx[label_str] = len(self.fine_grained_classes)
                            self.fine_grained_classes.append(label_str)
                    except (ValueError, IndexError):
                        continue
            
            if is_valid_file:
                valid_files.append(img_path)
        
        self.image_files = valid_files

    def _fix_labels(self):
        '''
        Fixes specific labels in the fine-grained class map.
        Changes 'ne' to 'neq', 'ge' to 'geq', 'le' to 'leq' and 'exist' to 'exists'.
        '''
        for i, cls_name in enumerate(self.fine_grained_classes):
            if cls_name == 'ne':
                self.fine_grained_classes[i] = 'neq'
                self.fine_grained_to_idx['neq'] = self.fine_grained_to_idx.pop('ne')
            elif cls_name == 'ge':
                self.fine_grained_classes[i] = 'geq'
                self.fine_grained_to_idx['geq'] = self.fine_grained_to_idx.pop('ge')
            elif cls_name == 'le':
                self.fine_grained_classes[i] = 'leq'
                self.fine_grained_to_idx['leq'] = self.fine_grained_to_idx.pop('le')
            elif cls_name == '<':
                self.fine_grained_classes[i] = 'lt'
                self.fine_grained_to_idx['lt'] = self.fine_grained_to_idx.pop('<')
            elif cls_name == '>':
                self.fine_grained_classes[i] = 'gt'
                self.fine_grained_to_idx['gt'] = self.fine_grained_to_idx.pop('>')
            elif cls_name == 'exist':
                self.fine_grained_classes[i] = 'exists'
                self.fine_grained_to_idx['exists'] = self.fine_grained_to_idx.pop('exist')


    def map_to_metaclass(self, label_str):
        '''
        Maps a specific symbol string to its meta-class ID.
        '''
        # digits
        if label_str.isdigit(): 
            return self.meta_class_to_idx['Digit']
            
        # latin letters
        if label_str.isalpha() and len(label_str) == 1:
            return self.meta_class_to_idx['Letter']
            
        # greek letters
        greek_vars = [
            '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\zeta', '\\eta', 
            '\\theta', '\\iota', '\\kappa', '\\lambda', '\\mu', '\\nu', '\\xi', 
            '\\pi', '\\rho', '\\sigma', '\\tau', '\\phi', '\\chi', '\\psi', '\\omega',
            '\\Delta', '\\Gamma', '\\Theta', '\\Lambda', '\\Xi', '\\Pi', '\\Sigma', '\\Phi', '\\Psi', '\\Omega'
        ]
        if label_str in greek_vars:
             return self.meta_class_to_idx['Letter']

        # operators (everything else, e.g., +, -, =, <, >, trigonometric functions, etc.)
        return self.meta_class_to_idx['Operator']

    def parse_lg_file(self, lg_path):
        '''
        Parses an LG file to extract bounding boxes and labels.
        Expected format per line: ['BB', '<label>_<instance_id>', x_min, y_min, x_max, y_max]
        Returns: (boxes, fine_labels, meta_labels)
        '''
        boxes = []
        fine_labels = []
        meta_labels = []
        
        with open(lg_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if not row or row[0] != 'BB': continue

                try:
                    full_label_str = row[1].strip()
                    label_str = full_label_str.rsplit('_', 1)[0]
                    
                    # skip unknown labels (or those we excluded during init)
                    if label_str not in self.fine_grained_to_idx:
                        continue
                        
                    fine_id = self.fine_grained_to_idx[label_str]
                    meta_id = self.map_to_metaclass(label_str)
                    
                    x1 = float(row[2])
                    y1 = float(row[3])
                    x2 = float(row[4])
                    y2 = float(row[5])

                    width = x2 - x1
                    height = y2 - y1

                    if width < 1.0 or height < 1.0:
                        x2 = x1 + max(width, 1.0)
                        y2 = y1 + max(height, 1.0)

                    boxes.append([x1, y1, x2, y2])
                    fine_labels.append(fine_id)
                    meta_labels.append(meta_id)
                    
                except (ValueError, IndexError):
                    continue
        
        # handle empty files
        if len(boxes) == 0:
            return (torch.zeros((0, 4), dtype=torch.float32), 
                    torch.zeros((0,), dtype=torch.int64),
                    torch.zeros((0,), dtype=torch.int64))
            
        return (torch.tensor(boxes, dtype=torch.float32), 
                torch.tensor(fine_labels, dtype=torch.int64), 
                torch.tensor(meta_labels, dtype=torch.int64))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        lg_path = img_path.replace('.png', '.lg')
        
        if not os.path.exists(lg_path):
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            fine_labels = torch.zeros((0,), dtype=torch.int64)
            meta_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes, fine_labels, meta_labels = self.parse_lg_file(lg_path)

        target = {}
        target['boxes'] = boxes
        target['image_id'] = torch.tensor([idx])
        
        # metaclass
        target['labels'] = meta_labels 

        # specific symbol labels        
        target['specific_labels'] = fine_labels

        if self.transform:
            image = self.transform(image)
            
        return image, target
    

class TrainingSymbolsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.folder_names = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # remove the folder named "dot" (garbage data)
        self.folder_names = [d for d in self.folder_names if d != 'dot']
        
        self.class_names = [d.replace('_', '') for d in self.folder_names]
        
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        self.samples = []
        for i, folder_name in enumerate(self.folder_names):
            cls_folder = os.path.join(root_dir, folder_name)
            image_paths = glob.glob(os.path.join(cls_folder, '*.png'))
            cls_idx = i 
            for img_path in image_paths:
                self.samples.append((img_path, cls_idx))
                
        self.classes = self.class_names
        # save classes in a file for the transformer to use later
        with open('data/symbol_classes.txt', 'w') as f:
            for cls_name in self.class_names:
                f.write(f'{cls_name}\n')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label_idx, dtype=torch.long)

    def get_class_name(self, label_idx):
        return self.classes[label_idx]


def collate_fn(batch):
    return tuple(zip(*batch))

def get_detector_transforms(train=True):
    transforms = []
    
    transforms.append(T.ToTensor())
    
    if train:
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))
        transforms.append(T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5))

    return T.Compose(transforms)

def get_classifier_transforms(train=True):
    transforms = []
    
    transforms.append(T.Resize((32, 32)))
    
    if train:
        transforms.append(T.RandomRotation(degrees=10, fill=1)) # fill=1 assumes white background
        
        transforms.append(T.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1), # Shift image by 10%
            scale=(0.8, 1.2),     # Zoom in/out
            shear=10,             # Slant
            fill=1                # White background padding
        ))
        
        transforms.append(T.RandomPerspective(distortion_scale=0.2, p=0.5))

    transforms.append(T.ToTensor())

    # ImageNet normalization    
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)


if __name__ == '__main__':
    transform = T.Compose([
        T.ToTensor()
    ])

    expression_dataset = FullExpressionsDataset(root_dir='dataset/CROHME/FullExpressionsDataset/CROHME2019_train_png', transform=transform)
    symbol_dataset = TrainingSymbolsDataset(root_dir='dataset/CROHME/TrainingSymbolsDataset/', transform=transform)
    
    expression_data_loader = DataLoader(
        expression_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    symbol_data_loader = DataLoader(
        symbol_dataset, 
        batch_size=2, 
        shuffle=True
    )

    images, targets = next(iter(expression_data_loader))
    
    print(f'\nBatch Info:')
    print(f'Image Shape: {images[0].shape}')
    print(f'Target Keys: {targets[0].keys()}')
    print(f'Boxes: \n{targets[0]['boxes']}')
    print(f'Meta Labels (0-3): {targets[0]['labels']}')
    print(f'Specific Labels (0-N): {targets[0]['specific_labels']}')
    
    inv_map = {v: k for k, v in expression_dataset.fine_grained_to_idx.items()}
    inv_meta = {v: k for k, v in expression_dataset.meta_class_to_idx.items()}
    
    print('\nLabel Translation:')
    for i, specific_id in enumerate(targets[0]['specific_labels']):
        s_name = inv_map[specific_id.item()]
        m_id = targets[0]['labels'][i].item()
        m_name = inv_meta[m_id]
        print(f"  Symbol '{s_name}' mapped to Meta-Class '{m_name}'")

    # visualize the expression image and its boxes
    img = images[0].permute(1, 2, 0).numpy()
    plt.imshow(img)
    ax = plt.gca()
    for box in targets[0]['boxes']:
        x1, y1, x2, y2 = box.numpy()
        print(f'Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}')
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                             fill=False, color='red', linewidth=2)
        # add both meta and specific labels
        idx = (targets[0]['boxes'] == box).all(dim=1).nonzero(as_tuple=True)[0].item()
        specific_id = targets[0]['specific_labels'][idx].item()
        meta_id = targets[0]['labels'][idx].item()
        specific_name = inv_map[specific_id]
        meta_name = inv_meta[meta_id]
        ax.text(x1, y1 - 10, f'{meta_name} ({specific_name})', 
                color='blue', fontsize=8, weight='bold')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()
    
    images, labels = next(iter(symbol_data_loader))
    print(f'Image Shape: {images[0].shape}')
    print(f'Label: {symbol_dataset.get_class_name(labels[0].item())}')
    plt.imshow(images[0].permute(1, 2, 0))
    plt.axis('off')
    plt.show()

    # find expression classes which are not in symbol classes
    expression_classes = set(expression_dataset.fine_grained_classes)
    symbol_classes = set(symbol_dataset.classes)
    print(f'\nTotal expression classes: {len(expression_classes)}')
    print(f'Total symbol classes: {len(symbol_classes)}')
    missing_classes = expression_classes - symbol_classes
    print(f'\nExpression classes not found in symbol dataset: {missing_classes}')