import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


def load_vocab(filepath):
    with open(filepath, 'r') as f:
        # Strip newlines and spaces
        return [line.strip() for line in f if line.strip()]

VOCAB = load_vocab('data/symbol_classes.txt')

LATEX_MAP = {
    # Greeks
    'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma', 'delta': '\\delta',
    'theta': '\\theta', 'phi': '\\phi', 'pi': '\\pi', 'sigma': '\\sigma',
    'lambda': '\\lambda', 'mu': '\\mu', 'Delta': '\\Delta',
    # Functions
    'sin': '\\sin', 'cos': '\\cos', 'tan': '\\tan', 
    'log': '\\log', 'lim': '\\lim',
    # Operators
    'infty': '\\infty', 'int': '\\int', 'sum': '\\sum', 'sqrt': '\\sqrt',
    'pm': '\\pm', 'geq': '\\geq', 'leq': '\\leq', 'neq': '\\neq',
    'rightarrow': '\\rightarrow', 'times': '\\times', 'div': '\\div',
    # Special
    '{': '\\{', '}': '\\}'  # Escaping braces if they are literal characters
}


class MathNode:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.rel_x = 0  # Relative to parent
        self.rel_y = 0
        self.children = []
        self.class_name = ''

    def compute_layout(self):
        '''Recursively calculate size of this node based on children'''
        pass

    def get_boxes(self, abs_x, abs_y):
        '''Recursively get absolute coordinates for all atoms'''
        boxes = []
        # if the node is a class (atom), add its box
        if self.class_name:
            # [class_name, x_center, y_center, width, height]
            boxes.append({
                'class': self.class_name,
                'box': [abs_x + self.rel_x, abs_y + self.rel_y, self.width, self.height]
            })
        
        # Recurse for children
        for child in self.children:
            boxes.extend(child.get_boxes(abs_x + self.rel_x, abs_y + self.rel_y))
        return boxes

    def get_latex(self):
        '''Recursively build the LaTeX string'''
        return ''


class Atom(MathNode):
    '''Basic character node (e.g., 'x', '1', '+')'''
    def __init__(self, char):
        super().__init__()
        self.class_name = char
        # generic size for a character
        if len(char) > 1 and char.startswith('\\'):   # e.g., \sin, \cos
            self.width = 3.0
        else:
            self.width = 1.0
        self.height = 1.0

    def compute_layout(self):
        pass 

    def get_latex(self):
        return LATEX_MAP.get(self.class_name, self.class_name)


class Row(MathNode):
    '''
    A horizontal sequence of MathNodes.
    E.g., for 'xy+1', it's a Row of [Atom('x'), Atom('y'), Atom('+'), Atom('1')]
    '''
    def __init__(self, items):
        super().__init__()
        self.children = items

    def compute_layout(self):
        current_x = 0
        max_h = 0
        padding = 0.2  # space between chars
        
        for child in self.children:
            child.compute_layout()
            child.rel_x = current_x + (child.width / 2)  # center it
            child.rel_y = 0
            
            current_x += child.width + padding
            max_h = max(max_h, child.height)
            
        self.width = current_x
        self.height = max_h

    def get_latex(self):
        return ' '.join([c.get_latex() for c in self.children])

class Fraction(MathNode):
    '''A fraction node with numerator and denominator.'''
    def __init__(self, num, den):
        super().__init__()
        self.numerator = num
        self.denominator = den
        
        self.bar = Atom('frac_bar')  # fraction bar
        self.children = [self.numerator, self.bar, self.denominator]

    def compute_layout(self):
        self.numerator.compute_layout()
        self.denominator.compute_layout()
        
        width = max(self.numerator.width, self.denominator.width)
        
        # numerator
        self.numerator.rel_x = width / 2
        self.numerator.rel_y = 1.0 + (self.numerator.height / 2)
        
        # fraction bar
        self.bar.width = width
        self.bar.height = 0.1
        self.bar.rel_x = width / 2
        self.bar.rel_y = 0
        
        # denominator
        self.denominator.rel_x = width / 2
        self.denominator.rel_y = -1.0 - (self.denominator.height / 2)
        
        self.width = width
        self.height = self.numerator.height + self.denominator.height + 2.0

    def get_latex(self):
        return f'\\frac{{{self.numerator.get_latex()}}}{{{self.denominator.get_latex()}}}'
    

class Superscript(MathNode):
    def __init__(self, base, exp):
        super().__init__()
        self.base = base
        self.exp = exp
        self.children = [self.base, self.exp]

    def compute_layout(self):
        self.base.compute_layout()
        self.exp.compute_layout()
        
        # scale down the exponent
        scale_factor = 0.7
        self.exp.width *= scale_factor
        self.exp.height *= scale_factor
        
        self.base.rel_x = 0
        self.base.rel_y = 0
        
        self.exp.rel_x = (self.base.width / 2) + (self.exp.width / 2)
        self.exp.rel_y = (self.base.height * 0.6) 
        
        self.width = self.base.width + self.exp.width
        # max of base bottom and exp top
        self.height = max(self.base.height, self.exp.height + self.exp.rel_y)

    def get_latex(self):
        return f'{self.base.get_latex()}^{{{self.exp.get_latex()}}}'


def generate_random_tree(depth=0, max_depth=3):
    '''Recursively builds a math expression tree including Trig and Powers'''
    
    # Base Case: Stop recursion
    if depth >= max_depth or random.random() > 0.8:
        # 50% chance of single atom, 50% chance of short row
        if random.random() > 0.5:
            return Atom(random.choice(VOCAB))
        else:
            return Row([Atom('x'), Atom('+'), Atom('1')]) # Simple filler
    
    # Recursive Case: Choose a structure
    choice = random.choice(['frac', 'row', 'super', 'trig'])
    
    if choice == 'frac':
        num = generate_random_tree(depth + 1, max_depth)
        den = generate_random_tree(depth + 1, max_depth)
        return Fraction(num, den)
        
    elif choice == 'super':
        # Base is often simple (x or y), Exponent is simple (2 or n)
        # But we can allow nesting! (e.g. sin^2)
        base = generate_random_tree(depth + 1, max_depth)
        exp = generate_random_tree(depth + 1, max_depth)
        return Superscript(base, exp)
    
    elif choice == 'trig':
        # Select from the loaded VOCAB (e.g., 'sin', not '\sin')
        # We filter VOCAB to find the trig functions available in your file
        available_trigs = [t for t in ['sin', 'cos', 'tan'] if t in VOCAB]
        func_name = random.choice(available_trigs)
        
        arg = generate_random_tree(depth + 1, max_depth)
        return Row([Atom(func_name), arg])
        
    elif choice == 'row':
        items = []
        for _ in range(random.randint(2, 3)):
            items.append(generate_random_tree(depth + 1, max_depth))
        return Row(items)


class SyntheticMathDataset(Dataset):
    def __init__(self, vocab_list, num_samples=1000, max_boxes=50, jitter_range=(0.05, 0.35)):
        self.vocab_list = vocab_list
        self.num_samples = num_samples
        self.max_boxes = max_boxes
        self.jitter_range = jitter_range
        
        # Create a mapping from class_name to Integer ID
        self.class_map = {k: i for i, k in enumerate(self.vocab_list + ['frac_bar', '<sos>', '<eos>', '<PAD>'])}
        self.vocab_size = len(self.class_map)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Generate Expression
        root = generate_random_tree(max_depth=3)
        root.compute_layout()
        
        # 2. Get Perfect Boxes
        raw_boxes = root.get_boxes(0, 0)
        latex_str = root.get_latex()
        
        # 3. Process into Tensors
        input_vectors = []
        
        # --- DYNAMIC JITTER LOGIC ---
        # Pick a random 'messiness' for THIS specific equation.
        # Some samples will be neat (0.05), others will be messy (0.35)
        current_jitter = np.random.uniform(self.jitter_range[0], self.jitter_range[1])
        
        for item in raw_boxes:
            cls_id = self.class_map.get(item['class'], 0)
            b = item['box'] # [x, y, w, h]
            
            nx = b[0] + np.random.uniform(-current_jitter, current_jitter)
            ny = b[1] + np.random.uniform(-current_jitter, current_jitter)
            
            scale_w = np.random.uniform(1.0 - current_jitter, 1.0 + current_jitter)
            scale_h = np.random.uniform(1.0 - current_jitter, 1.0 + current_jitter)
            
            nw = b[2] * scale_w
            nh = b[3] * scale_h
            
            input_vectors.append([cls_id, nx, ny, nw, nh])
            
        # 4. Padding
        actual_len = len(input_vectors)
        if actual_len > self.max_boxes:
            input_vectors = input_vectors[:self.max_boxes]
        else:
            while len(input_vectors) < self.max_boxes:
                input_vectors.append([self.class_map['<PAD>'], 0, 0, 0, 0])
                
        # 5. Return
        return {
            'input_boxes': torch.tensor(input_vectors, dtype=torch.float32),
            'target_latex': latex_str,
            'length': actual_len
        }


if __name__ == '__main__':
    ds = SyntheticMathDataset(num_samples=10)
    dl = DataLoader(ds, batch_size=2)
    
    sample = next(iter(dl))
    
    print('--- Generated Batch ---')
    print('Input Tensor Shape:', sample['input_boxes'].shape) # (Batch, Max_Boxes, 5)
    print('\nSample 1 LaTeX:', sample['target_latex'][0])
    print('Sample 1 First Box (Class, X, Y, W, H):')
    print(sample['input_boxes'][0][0])
    
    # Example Output:
    # LaTeX: \frac{x}{y+1}
    # Box: [ClassID, 0.5, 1.2, 1.0, 1.0] <- 'x' roughly above 'y'