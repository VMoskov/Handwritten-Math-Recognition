import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    '''
    Standard sinusoidal positional encoding for the output LaTeX sequence.
    '''
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Seq_Len, Batch, Dim)
        return x + self.pe[:, :x.size(1), :]

class TranslatorTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=4, num_layers=3, max_seq_len=100):
        super().__init__()
        
        # --- 1. The "Fusion" Embedding ---
        # We split the model dimension: Half for the Class, Half for the Box geometry
        self.d_model = d_model
        
        # Embed the Class ID (from your Classifier)
        self.class_embedding = nn.Embedding(num_classes, d_model // 2)
        
        # Project the Coordinates [x, y, w, h] -> Vector
        self.box_projection = nn.Linear(4, d_model // 2)
        
        # --- 2. The Encoder (Understanding Spatial Layout) ---
        # Note: We do NOT use Positional Encoding here because the "position" 
        # is already encoded in the x,y coordinates themselves.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. The Decoder (Generating LaTeX) ---
        # This part looks like a standard language model
        self.decoder_embedding = nn.Embedding(num_classes, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # --- 4. Output Head ---
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src_boxes, tgt_tokens, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        '''
        Args:
            src_boxes: (Batch, N_Boxes, 5) tensor. 
                       Last dim is [class_id, x, y, w, h].
            tgt_tokens: (Batch, Seq_Len) tensor of current LaTeX tokens (for training).
        '''
        
        # --- A. Prepare Source Embeddings ---
        # Extract components
        src_cls = src_boxes[:, :, 0].long()     # (Batch, N)
        src_coords = src_boxes[:, :, 1:].float() # (Batch, N, 4)
        
        # Embed and Concatenate
        emb_cls = self.class_embedding(src_cls)          # (Batch, N, d_model/2)
        emb_coords = self.box_projection(src_coords)     # (Batch, N, d_model/2)
        src_emb = torch.cat([emb_cls, emb_coords], dim=2) # (Batch, N, d_model)
        
        # --- B. Run Encoder ---
        # The encoder learns the relationship between "Superscript 2" and "Base x"
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # --- C. Run Decoder ---
        # Standard auto-regressive decoding
        tgt_emb = self.decoder_embedding(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.decoder(
            tgt_emb, 
            memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # --- D. Project to Vocabulary ---
        logits = self.fc_out(output)
        
        return logits
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generates an upper triangular Boolean causal mask.
        True = MASKED (do not attend).
        False = UNMASKED (attend).
        """
        return torch.triu(torch.full((sz, sz), True), diagonal=1)


def get_translator_model(num_classes, weights_path=None, device=None):
    '''
    Builds the Box-to-LaTeX Transformer.
    
    Args:
        num_classes (int): Total unique tokens (must match your symbol_classes.txt).
        weights_path (str, optional): Path to .pth file to load trained weights.
        device (torch.device, optional): Device to load weights onto.
        
    Returns:
        model (torch.nn.Module): The transformer model.
    '''
    model = TranslatorTransformer(
        num_classes=num_classes,
        d_model=256,
        nhead=4,
        num_layers=3
    )
    
    if weights_path:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print(f'Loading translator weights from {weights_path}...')
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    
    return model