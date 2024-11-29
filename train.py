import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import time
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
import math
from datetime import datetime
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    device = torch.device(f'cuda:{ddp_local_rank}')
    torch.cuda.set_device(device)
    master = ddp_rank == 0

else:
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    master = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using {device}")

#-----------------------------------DDP----------------------------------------------------------
class AdamMini(Optimizer):

    def __init__(self, model, vocab_size, n_heads, lr=0.00059, beta1=0.9, beta2=0.999,
                 weight_decay=0.0, eps=1e-8):
        self.model = model
        self.vocab_size = vocab_size 
        self.n_heads = n_heads
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.state_dict_custom = self._init_state() 

        super(AdamMini, self).__init__(model.parameters(), {})

    def _init_state(self):
        state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                state[name] = {
                    'm': torch.zeros_like(param, memory_format=torch.preserve_format),  
                    'v': torch.zeros_like(param, memory_format=torch.preserve_format), 
                    'step': 0
                }
        return state

    def _update_param_block(self, param_data, grad, m, v):
        m_new = (1 - self.beta1) * grad + self.beta1 * m
        corrected_beta1 = 1 - (self.beta1 ** (self.step_count + 1))
        momentum = m_new / corrected_beta1

        v_new = (1 - self.beta2) * (grad * grad).mean() + self.beta2 * v
        corrected_beta2 = 1 - (self.beta2 ** (self.step_count + 1))
        velocity = v_new / corrected_beta2

        update = momentum / (torch.sqrt(velocity) + self.eps)
        param_data.add_(-self.lr * update)
        if self.weight_decay > 0:
            param_data.add_(-self.lr * self.weight_decay * param_data)
            
        return m_new, v_new

    @torch.no_grad()
    def step(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            state = self.state_dict_custom[name]
            
            if 'emb' in name or ('output' in name and 'output_rms' not in name):
                for i in range(self.vocab_size):
                    grad = param.grad[i] if param.grad.dim() == 1 else param.grad[i, :]
                    m = state['m'][i] if state['m'].dim() == 1 else state['m'][i, :]
                    v = state['v'][i] if state['v'].dim() == 1 else state['v'][i, :]
                    
                    m_new, v_new = self._update_param_block(param.data[i], grad, m, v)
                    
                    if state['m'].dim() == 1:
                        state['m'][i] = m_new
                        state['v'][i] = v_new
                    else:
                        state['m'][i, :] = m_new
                        state['v'][i, :] = v_new
            
            elif 'query' in name or 'key' in name:
                head_size = param.shape[-1] // self.n_heads
                for i in range(self.n_heads):
                    slice_idx = slice(i * head_size, (i + 1) * head_size)
                    grad = param.grad[slice_idx]
                    m, v = state['m'][slice_idx], state['v'][slice_idx]
                    
                    m_new, v_new = self._update_param_block(
                        param.data[slice_idx], grad, m, v
                    )
                    state['m'][slice_idx] = m_new
                    state['v'][slice_idx] = v_new
            
            elif any(x in name for x in ['value', 'proj', 'mlp']):
                for i in range(param.shape[-1]):
                    grad = param.grad[:, i]
                    m, v = state['m'][:, i], state['v'][:, i]
                    
                    m_new, v_new = self._update_param_block(
                        param.data[:, i], grad, m, v
                    )
                    state['m'][:, i] = m_new
                    state['v'][:, i] = v_new
            
            else:
                m_new, v_new = self._update_param_block(
                    param.data, param.grad, state['m'], state['v']
                )
                state['m'] = m_new
                state['v'] = v_new
            
            state['step'] += 1
        
        self.step_count += 1

    def state_dict(self):
        for state in self.state_dict_custom.values():
            dist.all_reduce(state['m'], op=dist.ReduceOp.SUM)
            dist.all_reduce(state['v'], op=dist.ReduceOp.SUM)
            state['m'] /= dist.get_world_size()
            state['v'] /= dist.get_world_size()
        return self.state_dict_custom

    def load_state_dict(self, state_dict):
        self.state_dict_custom = state_dict

class Config:
    vocab_size = 50304
    d_model = 1024
    batch_size = 512
    context_length = 32
    n_heads = 8
    n_layers = 8
    learning_rate = 0.0001
    weight_decay = 1e-5
    max_epochs = 3000

config = Config()

class Rotary(nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        assert x.ndim == 4 
        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)

# arXiv:2104.09864v5
class RMSNorm(nn.Module):
    def __init__(self, config) -> None:
        super(RMSNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(config.d_model)) 
        self.alpha = 0.99
        self.eps = 1e-8

    def forward(self, x):
        B, T, C = x.shape
        rms = (torch.mean(x**2, dim=-1, keepdim=True))**0.5
        x = x / (rms + self.eps)
        x = self.g * x
        return x

# arXiv:2002.05202v1
class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model) -> None:
        super(FFN_SwiGLU, self).__init__()
        self.w1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.w2 = nn.Linear(d_model * 4, d_model, bias=False)
        self.beta = nn.Parameter(torch.ones(1))
        self.v = nn.Linear(d_model, d_model * 4, bias=False)

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
        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model)
        self.wo.flag = 1
        self.rotary = Rotary(config.d_model // config.n_heads)
        self.wo.weight.data.zero_()

    def forward(self, x):
        B, T, C = x.size()

        q = self.wq(x).view(B, T, self.n, self.h).transpose(1, 2)  # (B, n, T, h)
        k = self.wk(x).view(B, T, self.n, self.h).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n, self.h).transpose(1, 2)

        q, k = self.rotary(q), self.rotary(k)
        # Flash Attention
        att_val = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        # Concatenate multiple heads
        attention = att_val.transpose(1, 2).contiguous().view(B, T, C)
        final = self.wo(attention)
        return final

class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(config.d_model, config.d_model * 4, bias=False)
        self.w2 = nn.Linear(config.d_model * 4, config.d_model, bias=False)
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
        self.final.weight.data.zero_()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # for every residual connection we scale the weights by 1/sqrt(2*n_layers) so as to prevent the variance go boom
            if hasattr(module, 'flag') and module.flag == 1:
                std *= (2 * self.config.n_layers) ** -0.5
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x, targets=None):
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
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    model_clone = model.module
else:
    model_clone = model

optimizer = AdamMini(
    model_clone,
    config.vocab_size,
    config.n_heads,
    lr=config.learning_rate,      
    weight_decay=config.weight_decay 
)

warmup_steps = 1000  
max_steps = 3000     
max_lr = config.learning_rate  
min_lr = 0.0         

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_rate = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_rate <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_rate)) 
    return min_lr + coeff * (max_lr - min_lr)

def train(model, opt, config, epochs=3000, grad_accum_steps=2):
    losses = []
    log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log_metrics(epoch, train_loss, val_loss=None, tokens_per_sec=None, mfu=None, grad_norm=None):
        if master:
            with open(log_file, 'a') as f:
                log_line = f"Step {epoch} | Train loss: {train_loss:.4f}"
                if val_loss is not None:
                    log_line += f" | Val loss: {val_loss:.4f}"
                if tokens_per_sec is not None:
                    log_line += f" | Tokens/sec: {tokens_per_sec:.2f}"
                if grad_norm is not None:
                    log_line += f" | Grad norm: {grad_norm:.4f}"
                f.write(log_line + "\n")
                print(log_line)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        
        cumulative_loss = torch.zeros(1, device=device)
        start_time = time.time()

        for step in range(grad_accum_steps):
            x, y = get_batch('train')
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                y_pred, loss = model(x, y)
            loss = loss / grad_accum_steps
            cumulative_loss += loss.detach()
            torch.cuda.synchronize()

            if ddp:
                model_clone.require_backward_grad_sync = step == grad_accum_steps - 1
            loss.backward()

        grad_norm = torch.zeros(1, device=device)
        for p in model_clone.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm.sqrt()

        if ddp:
            dist.all_reduce(cumulative_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(grad_norm, op=dist.ReduceOp.AVG)
            torch.distributed.barrier()

        torch.cuda.synchronize()
        opt.step()

        current_step = epoch * grad_accum_steps + step
        for param_group in opt.param_groups:
            param_group['lr'] = get_lr(current_step)

        tokens_per_batch = config.batch_size * config.context_length * grad_accum_steps
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_per_batch / elapsed

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_x, val_y = get_batch('val')
                val_x, val_y = val_x.to(device), val_y.to(device)
                _, val_loss = model(val_x, val_y)
                if ddp:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                    torch.distributed.barrier()
            
            log_metrics(
                epoch=epoch,
                train_loss=cumulative_loss.item(),
                val_loss=val_loss.item(),
                tokens_per_sec=tokens_per_sec,
                grad_norm=grad_norm.item()
            )
            
            losses.append({
                'train': cumulative_loss.item(),
                'val': val_loss.item(),
                'tokens_per_sec': tokens_per_sec,
                'grad_norm': grad_norm.item()
            })

        torch.cuda.empty_cache()  

    return pd.DataFrame(losses).plot()

data_dir = "./fineweb10B"

def get_batch(split):
    if split == 'train':
        file_path = os.path.join(data_dir, "train.bin")
    else:
        file_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist. Please ensure you've combined the binary files.")
    
    dtype = np.uint16  
    data = np.memmap(file_path, dtype=dtype, mode='r')

    if len(data) < config.context_length + 1:
        raise ValueError(f"File {file_path} is too small for context length {config.context_length}")

    ix = torch.randint(0, len(data) - config.context_length - 1, (config.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+config.context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+config.context_length].astype(np.int64)) for i in ix])

    # if master:
    #     print(f"Batch min value: x={x.min().item()}, y={y.min().item()}")
    #     print(f"Batch max value: x={x.max().item()}, y={y.max().item()}")
    # Verification
    # if torch.any(x >= config.vocab_size) or torch.any(y >= config.vocab_size):
    #     raise ValueError("Token index out of bounds in the batch. Please check your data preprocessing.")

    if device.type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

# def verify_data_loading():
#     print("Verifying data loading...")
#     x, y = get_batch('train')
#     print(f"Training batch shape - x: {x.shape}, y: {y.shape}")
#     print(f"Training data sample values - min: {x.min()}, max: {x.max()}")

# verify_data_loading()

# After training, save the model in a popular format
def save_model(model, model_name="my_model.pt"):
    torch.save(model.state_dict(), model_name) 

try:
    train(model, optimizer, config)
    save_model(model_clone, model_name="adamGpt.pt")  
    if "out of memory" in str(e):
        print("CUDA out of memory. Try reducing batch size or model size.")
    raise e
except Exception as e:
    print(f"Training failed: {e}")

# Destroy the process group if using DDP
if ddp:
    destroy_process_group()

