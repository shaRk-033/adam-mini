import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
import tiktoken
from adam_mini import Adam_mini
import math
from datetime import datetime
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# with open('input.txt', 'r') as f:
#     data = f.read()

# vocab = sorted(list(set(data)))
# encode = {char: i for i, char in enumerate(vocab)}
# decode = {i: char for i, char in enumerate(vocab)}
# enc_data = [encode[char] for char in data]
# ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
# tokenizer = tiktoken.get_encoding('gpt2')

class miniDataLoader:
    def __init__(self, config, data, process_rank, num_processes, is_train=True) -> None:
        self.bs = config.batch_size
        self.length = config.context_length
        self.data = data
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.is_train = is_train
        self.valid_indices = len(data) - self.length - 1
        
        # for distributed training, split indices among processes
        self.indices_per_process = self.valid_indices // num_processes
        self.start_idx = self.indices_per_process * process_rank
        self.end_idx = self.start_idx + self.indices_per_process
        
        self.current_idx = self.start_idx
        self.indices = None
        self.shuffle_data()
        
    def shuffle_data(self):
        if self.is_train:
            indices = torch.arange(self.start_idx, self.end_idx)
            self.indices = indices[torch.randperm(len(indices))]
        else:
            self.indices = torch.arange(self.start_idx, self.end_idx)
        self.current_idx = 0
    
    def get_data(self):
        B = self.bs
        T = self.length
        
        if self.current_idx + B >= len(self.indices):
            self.shuffle_data()
            
        batch_starts = self.indices[self.current_idx:self.current_idx + B]
        
        x = torch.stack([self.data[i:i+T] for i in batch_starts])
        y = torch.stack([self.data[i+1:i+T+1] for i in batch_starts])
        
        self.current_idx += B
        return x, y

#------------------------------------------------------------------------------------------------

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:

    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get('RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master = ddp_rank == 0

else:
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    master = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using {device}")

#-----------------------------------DDP----------------------------------------------------------

class Config:
    vocab_size = 50304
    d_model = 1024
    batch_size = 32
    context_length = 64
    n_heads = 8
    n_groups = 4
    n_layers = 8

config = Config()

# arXiv:2104.09864v5
class RMSNorm(nn.Module):
    def __init__(self, config) -> None:
        super(RMSNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(config.d_model)) 
        self.alpha = 0.99
        self.eps = 1e-8
    def forward(self, x):
        B, T, C = x.shape
        rms = (torch.mean(x**2, dim = -1, keepdim = True))**0.5
        x = x / (rms + self.eps)
        x = self.g * x
        return x

# arXiv:2002.05202v1
class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model) -> None:
        super(FFN_SwiGLU, self).__init__()
        self.w1 = nn.Linear(d_model, d_model*4, bias = False)
        self.w2 = nn.Linear(d_model * 4, d_model, bias = False)
        self.beta = nn.Parameter(torch.ones(1))
        self.v = nn.Linear(d_model, d_model * 4, bias = False)

    def forward(self, x):
        var1 = self.w1(x)
        var2 = self.v(x)
        swish = var1 * torch.sigmoid(var1 * self.beta)
        gate_out = swish * var2
        x = self.w2(gate_out)
        return x


class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

# vanilla attention and ROPE
class SelfAttn(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n = config.n_heads
        self.h = config.d_model // self.n
        self.q = nn.Linear(config.d_model, config.d_model, bias = False)
        self.k = nn.Linear(config.d_model, config.d_model, bias = False)
        self.v = nn.Linear(config.d_model, config.d_model, bias = False)
        self.o = nn.Linear(config.d_model, config.d_model)
        self.o.flag = 1
        self.cos, self.sin = self.get_rotary_matrix(config)

    def get_rotary_matrix(self, config):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, config.d_model, 2, device=device).float() / config.d_model))
        position = torch.arange(0, config.context_length, device=device).float()
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()
        sin = sin.repeat(config.n_heads, 1)
        cos = cos.repeat(config.n_heads, 1)
        return cos, sin

    def forward(self, x):
        b,t,c = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        n = self.n
        h = self.h

        q_rotated = (q * self.cos) + (self.rotate_half(q) * self.sin)
        k_rotated = (k * self.cos) + (self.rotate_half(k) * self.sin)

        q = q_rotated.view(b, t, n, h).transpose(1, 2)
        k = k_rotated.view(b, t, n, h).transpose(1, 2)
        v = v.view(b, t, n, h).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p= 0, is_causal=True)

        attn = attn.transpose(1, 2).contiguous().view(b, t, c)
        return attn

    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x[...,0], x[...,1]
        return torch.cat((-x2, x1), dim=-1)

class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(config.d_model, config.d_model * 4, bias = False)
        self.w2 = nn.Linear(config.d_model * 4, config.d_model, bias = False)
        self.act = ReLUSquared()
        self.w2.flag = 1

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super(DecoderLayer, self).__init__()
        self.attn = SelfAttn(config)
        self.rmsn1 = RMSNorm(config)
        self.ffn = FFN(config)
        self.rmsn2 = RMSNorm(config)
    def forward(self, x):
        x1 = self.attn(self.rmsn1(x)) + x
        x2 = self.ffn(self.rmsn2(x1)) + x1
        return x2

class Gpt(nn.Module):
    def __init__(self, config):
        super(Gpt, self).__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.final_rms = RMSNorm(config)
        self.final = nn.Linear(config.d_model, config.vocab_size)
        print("Number of params: ", sum(p.numel() for p in self.parameters()))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # for every residual connection we scale the weights by 1/sqrt(2*n_layers) so as to prevent the variance go boom
            if hasattr(module, 'flag') and module.flag == 1:
                std *= (2 * self.config.n_layers) ** -0.5
            module.weight.data.normal_(mean = 0.0, std = std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean = 0.0, std = 0.02)

    def forward(self, x, targets = None):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_rms(x)
        x = self.final(x)
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, self.config.vocab_size), targets.view(-1))
            return x, loss
        else:
            return x
    def generate(self, start_prompt, tokenizer, max_length=100):
        self.eval()
        input_ids = tokenizer.encode(start_prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated = input_ids[:]
        for _ in range(max_length):
            x = input_tensor[:, -self.config.context_length:]
            logits = self(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)

        generated_text = tokenizer.decode(generated)
        return generated_text

model = Gpt(config).to(device)
model = torch.compile(model, fullgraph=True, mode='reduce-overhead')

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
model_clone = model.module if ddp else model

# opt = optim.Adam(model.parameters(), lr=0.001)
optimizer = Adam_mini(
            named_parameters = model.named_parameters(),
            lr = 0.001,
            betas = (0.9, 0.95),
            eps = 1e-8,
            dim = config.d_model,
            n_heads = config.n_heads,
            )

def calculate_mfu(model, config, tokens_per_sec):
    N = sum(p.numel() for p in model.parameters())
    L = config.n_layers
    H = config.d_model
    Q = config.context_length
    B = config.batch_size
    
    flops_per_token = 6*N + 12*L*H*Q
    flops_per_batch = flops_per_token * B
    flops_per_sec = flops_per_batch * tokens_per_sec
    theoretical_flops = torch.cuda.get_device_properties(0).multi_processor_count * 1e12  # A100 theoretical FLOPS
    mfu = flops_per_sec / theoretical_flops
    return mfu

def train(train_loader, val_loader, model, opt, config, epochs=3000, grad_accum_steps=4):
    losses = []
    log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log_metrics(epoch, train_loss, val_loss=None, tokens_per_sec=None, mfu=None, grad_norm=None):
        with open(log_file, 'a') as f:
            log_line = f"Epoch {epoch} | Train loss: {train_loss:.4f}"
            if val_loss is not None:
                log_line += f" | Val loss: {val_loss:.4f}"
            if tokens_per_sec is not None:
                log_line += f" | Tokens/sec: {tokens_per_sec:.2f}"
            if mfu is not None:
                log_line += f" | MFU: {mfu:.2%}"
            if grad_norm is not None:
                log_line += f" | Grad norm: {grad_norm:.4f}"
            f.write(log_line + "\n")
            print(log_line)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        
        cumulative_loss = torch.zeros(1, device=device)
        start_time = time.time()

        x, y = train_loader.get_data()
        x, y = x.to(device), y.to(device)

        for step in range(grad_accum_steps):
            y_pred, loss = model(x, y)
            loss = loss / grad_accum_steps
            cumulative_loss += loss.detach()

            if ddp:
                model_clone.require_backward_grad_sync = step == grad_accum_steps - 1
            loss.backward()

        if ddp:
            dist.all_reduce(cumulative_loss, op=dist.ReduceOp.AVG)

        grad_norm = torch.nn.utils.clip_grad_norm_(model_clone.parameters(), 1.0)
        opt.step()
        
        tokens_per_batch = config.batch_size * config.context_length
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_per_batch / elapsed
        
        mfu = calculate_mfu(model, config, tokens_per_sec)

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_x, val_y = val_loader.get_data()
                val_x, val_y = val_x.to(device), val_y.to(device)
                _, val_loss = model(val_x, val_y)
                if ddp:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            
            log_metrics(
                epoch=epoch,
                train_loss=cumulative_loss.item(),
                val_loss=val_loss.item(),
                tokens_per_sec=tokens_per_sec,
                mfu=mfu,
                grad_norm=grad_norm.item()
            )
            
            losses.append({
                'train': cumulative_loss.item(),
                'val': val_loss.item(),
                'tokens_per_sec': tokens_per_sec,
                'mfu': mfu,
                'grad_norm': grad_norm.item()
            })

        torch.cuda.synchronize()

    return pd.DataFrame(losses).plot()

def load_binary_files(directory, pattern="fineweb_train_*.bin", val_pattern="fineweb_val_*.bin"):
    def load_files(pattern):
        data = []
        for file_path in sorted(Path(directory).glob(pattern)):
            print(f"Loading {file_path.name}...")
            chunk = np.memmap(file_path, dtype=np.uint16, mode='r') 
            data.append(chunk)
        return np.concatenate(data) if data else None

    train_data = load_files(pattern)
    val_data = load_files(val_pattern)
    
    print(f"Loaded training data: {train_data.shape if train_data is not None else 'None'}")
    print(f"Loaded validation data: {val_data.shape if val_data is not None else 'None'}")
    
    return train_data, val_data

data_dir = "path/to/your/binary/files" 
train_data, val_data = load_binary_files(data_dir)

train_data = torch.from_numpy(train_data).long()
val_data = torch.from_numpy(val_data).long()

train_loader = miniDataLoader(config, train_data, ddp_rank, ddp_world_size, is_train=True)
val_loader = miniDataLoader(config, val_data, ddp_rank, ddp_world_size, is_train=False)


# def verify_data_loading():
#     print("Verifying data loading...")
#     x, y = train_loader.get_data()
#     print(f"Training batch shape - x: {x.shape}, y: {y.shape}")
#     print(f"Training data sample values - min: {x.min()}, max: {x.max()}")
    
#     x, y = val_loader.get_data()
#     print(f"Validation batch shape - x: {x.shape}, y: {y.shape}")
#     print(f"Validation data sample values - min: {x.min()}, max: {x.max()}")

# verify_data_loading()

try:
    train(train_loader, val_loader, model, optimizer, config)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("CUDA out of memory. Try reducing batch size or model size.")
    raise e
