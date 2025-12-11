"""
UstaModel - A simple transformer-based language model
"""

import torch
import torch.nn as nn
import os

# ============================================
# Model Architecture
# ============================================

def get_rotary_position_encoding(input: torch.Tensor, base=10000, device="cpu"):
    context_length, dimension = input.shape
    assert dimension % 2 == 0, "dimension must be even"
    half_dimension = dimension // 2
    freqs_indices = torch.arange(0, half_dimension, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (freqs_indices / dimension))
    positions = torch.arange(0, context_length, device=device, dtype=torch.float32).unsqueeze(1)
    angles = positions * freqs
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    input_even = input[:, :dimension // 2]
    input_odd = input[:, dimension // 2:]
    input_even_rotated = input_even * cos_angles - input_odd * sin_angles
    input_odd_rotated = input_even * sin_angles + input_odd * cos_angles
    input_rotated = torch.empty_like(input)
    input_rotated[:, :dimension // 2] = input_even_rotated
    input_rotated[:, dimension // 2:] = input_odd_rotated
    return input_rotated

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.get_pos = get_rotary_position_encoding

    def forward(self, x):
        x = self.embedding(x)
        x = self.get_pos(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, output_dim, context_length, num_heads, dropout_rate=0):
        super().__init__()
        self.context_length = context_length
        self.multi_head_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate)
        self.projection = nn.Linear(embedding_dim, output_dim)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        number_of_tokens = x.shape[0]
        x = x[:self.context_length]
        attention_mask = self.mask[:number_of_tokens, :number_of_tokens]
        out, _ = self.multi_head_attention(x, x, x, attn_mask=attention_mask)
        out = self.projection(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight * normalized_x

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
            )
        )

class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(embedding_dim, hidden_dim)
        self.up_proj = nn.Linear(embedding_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim)
        self.gelu = GELU()

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = self.gelu(gate)
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, embedding_dim, context_length, num_heads, dropout_rate=0.5)
        self.norm1 = LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

    def forward(self, x):
        res = self.norm1(x)
        x = self.self_attention(x)
        x = self.norm1(x)
        x = x + res
        res = self.norm2(x)
        x = self.mlp(x)
        x = self.norm2(x)
        x = x + res
        return x

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, context_length, num_layers):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, context_length)
        self.layers = nn.Sequential(
            *[DecoderBlock(embedding_dim, num_heads, context_length) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        x = self.lm_head(x)
        return x

# ============================================
# Vocabulary
# ============================================

VOCAB = {
    "the": 0, "capital": 1, "of": 2, "united": 3, "state": 4, "is": 5, "not": 6,
    "london": 7, "france": 8, "paris": 9, "and": 10, "berlin": 11, "germany": 12,
    "rome": 13, "in": 14, "italy": 15, "madrid": 16, "spain": 17, "lisbon": 18,
    "portugal": 19, "kingdom": 20, "washington": 21, "although": 22, "these": 23,
    "place": 24, "are": 25, "often": 26, "mention": 27, "together": 28, "each": 29,
    "country": 30, "has": 31, "its": 32, "own": 33, "identity": 34, "any": 35,
    "european": 36, "city": 37, "remain": 38, "important": 39, "with": 40, "a": 41,
    "rich": 42, "history": 43, "culture": 44, "europe": 45, "made": 46, "many": 47,
    "unique": 48, "world": 49, "while": 50, "known": 51, "for": 52, "art": 53,
    "fashion": 54, "famous": 55, "they": 56, "ed": 57, "s": 58, ".": 59, ",": 60,
    " ": 61, "<unk>": 62, "<pad>": 63, "+": 64, "-": 65, "1": 66, "2": 67, "3": 68,
    "4": 69, "5": 70, "6": 71, "7": 72, "8": 73, "9": 74, "buhari": 75, "hayz": 76,
    "quantum": 77, "computer": 78, "science": 79
}

REVERSE_VOCAB = {v: k for k, v in VOCAB.items()}

# Model hyperparameters
MODEL_CONFIG = {
    'vocab_size': len(VOCAB),
    'embedding_dim': 12,
    'num_heads': 4,
    'context_length': 32,
    'num_layers': 8
}

# ============================================
# Model Loading
# ============================================

def load_model(model_path='u_model.pth'):
    """Load the trained UstaModel"""
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(module_dir, model_path)
    
    torch.manual_seed(1)
    model = Model(
        vocab_size=MODEL_CONFIG['vocab_size'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        context_length=MODEL_CONFIG['context_length'],
        num_layers=MODEL_CONFIG['num_layers']
    )
    
    state_dict = torch.load(full_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

# ============================================
# Text Generation
# ============================================

def generate_text(model, start_text, max_tokens=50, temperature=1.0):
    """Generate text using the model"""
    model.eval()
    
    # Tokenize input
    tokens = []
    for word in start_text.lower().split():
        tokens.append(VOCAB.get(word, VOCAB["<unk>"]))
    
    if not tokens:
        tokens = [VOCAB["<unk>"]]
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Keep only context_length tokens
            input_ids = torch.tensor(generated[-MODEL_CONFIG['context_length']:], dtype=torch.long)
            
            # Get predictions
            outputs = model(input_ids)
            
            # Get last token prediction
            logits = outputs[-1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from distribution
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop if we hit padding
            if next_token == VOCAB["<pad>"]:
                break
            
            generated.append(next_token)
    
    # Decode
    generated_text = " ".join([REVERSE_VOCAB.get(token, "<unk>") for token in generated])
    return generated_text
