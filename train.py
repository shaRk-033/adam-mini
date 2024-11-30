import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import time
import os
import numpy as np
import pandas as pd
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
    master = ddp_rank == 0
    seed_offset = ddp_rank
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
    seed_offset = 0  # Define seed_offset for non-DDP

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(1337 + seed_offset)  # karpathy suggestion
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
        
        self.embd_names = {"embed", "embd", "wte"}
        self.output_names = {"lm_head", "output", "final"}
        self.wqk_names = {"k_proj", "q_proj", "wq", "wk", "query", "key"}
        self.wv_names = {"v_proj", "wv", "value"}
        self.attn_proj_names = {"o_proj", "wo", "attn.proj"}
        self.mlp_names = {"feed_forward", "linear", "mlp"}
        self.adam_block_names = {"bias"}
        
        self.state_dict_custom = self._init_state()
        super(AdamMini, self).__init__(model.parameters(), {})

    def _init_state(self):
        state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                state[name] = {
                    'm': torch.zeros_like(param, memory_format=torch.preserve_format),
                    'v': torch.zeros_like(param, memory_format=torch.preserve_format),
                    'step': 0,
                    'reduced': self._should_reduce_param(name)
                }
        return state
    
    def _should_reduce_param(self, name):
        if dist.is_initialized():
            if any(x in name.lower() for x in self.adam_block_names):
                return True
        return False

    def _update_adam_block(self, param, state, grad):
        state['v'].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        state['step'] += 1
        
        if self.weight_decay > 0:
            param.mul_(1 - self.lr * self.weight_decay)
            
        state['m'].lerp_(grad, 1 - self.beta1)
        
        bias_correction1 = 1 - self.beta1 ** state['step']
        bias_correction2 = 1 - self.beta2 ** state['step']
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        
        step_size = self.lr / bias_correction1
        h = (state['v'].sqrt() / bias_correction2_sqrt).add_(self.eps)
        
        param.addcdiv_(state['m'], h, value=-step_size)

    def _update_wqk_block(self, param, state, grad):
        head_size = param.shape[-1] // self.n_heads
        
        for i in range(self.n_heads):
            slice_idx = slice(i * head_size, (i + 1) * head_size)
            grad_head = grad[slice_idx]
            m, v = state['m'][slice_idx], state['v'][slice_idx]
            
            # Compute per-head statistics
            v_head = torch.mean(grad_head * grad_head)
            v.mul_(self.beta2).add_(v_head, alpha=1 - self.beta2)
            
            state['step'] += 1
            if self.weight_decay > 0:
                param.data[slice_idx].mul_(1 - self.lr * self.weight_decay)
            
            m.lerp_(grad_head, 1 - self.beta1)
            
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            bias_correction2_sqrt = math.sqrt(bias_correction2)
            
            h = (v.sqrt() / bias_correction2_sqrt).add_(self.eps)
            step_size = self.lr / bias_correction1
            
            param.data[slice_idx].addcdiv_(m, h, value=-step_size)

    def _update_neuron_block(self, param, state, grad):
        if grad.dim() <= 1:
            return self._update_adam_block(param, state, grad)
            
        neurons = grad.shape[0]
        for i in range(neurons):
            grad_neuron = grad[i]
            m, v = state['m'][i], state['v'][i]
            
            v_neuron = torch.mean(grad_neuron * grad_neuron)
            v.mul_(self.beta2).add_(v_neuron, alpha=1 - self.beta2)
            
            state['step'] += 1
            if self.weight_decay > 0:
                param.data[i].mul_(1 - self.lr * self.weight_decay)
            
            m.lerp_(grad_neuron, 1 - self.beta1)
            
            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']
            bias_correction2_sqrt = math.sqrt(bias_correction2)
            
            h = (v.sqrt() / bias_correction2_sqrt).add_(self.eps)
            step_size = self.lr / bias_correction1
            
            param.data[i].addcdiv_(m, h, value=-step_size)

    @torch.no_grad()
    def step(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            state = self.state_dict_custom[name]
            grad = param.grad
            
            if any(x in name.lower() for x in self.adam_block_names):
                self._update_adam_block(param, state, grad)
            elif any(x in name.lower() for x in self.wqk_names):
                self._update_wqk_block(param, state, grad)
            elif any(x in name.lower() for x in (self.embd_names | self.output_names | 
                    self.wv_names | self.mlp_names | self.attn_proj_names)):
                self._update_neuron_block(param, state, grad)
            else:
                self._update_adam_block(param, state, grad)
        
        self.step_count += 1

    def state_dict(self):
        if dist.is_initialized():
            for state in self.state_dict_custom.values():
                if state.get('reduced', False):
                    dist.all_reduce(state['m'])
                    dist.all_reduce(state['v'])
                    state['m'] /= dist.get_world_size()
                    state['v'] /= dist.get_world_size()
        return self.state_dict_custom

    def load_state_dict(self, state_dict):
        self.state_dict_custom = state_dict

class Config:
    vocab_size = 50304
    d_model = 768
    batch_size = 32
    context_length = 1024
    n_heads = 6
    n_layers = 8
    learning_rate = 4e-4
    weight_decay = 0.002
    max_epochs = 20000
    dropout = 0.0
    bias = False
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

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
        self.b = nn.Parameter(torch.zeros(config.d_model))
        self.alpha = 0.95
        self.eps = 1e-8

    def forward(self, x):
        B, T, C = x.shape
        rms = (torch.mean(x**2, dim=-1, keepdim=True))**0.5
        x = x / (rms + self.eps)
        x = self.g * x + self.b
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
        self.wq = nn.Linear(config.d_model, config.d_model)
        self.wk = nn.Linear(config.d_model, config.d_model)
        self.wv = nn.Linear(config.d_model, config.d_model)
        self.wo = nn.Linear(config.d_model, config.d_model)
        self.wo.flag = 1
        self.rotary = Rotary(config.d_model // config.n_heads)
        self.wo.weight.data.zero_()
        self.dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        q = self.wq(x).view(B, T, self.n, self.h).transpose(1, 2)  # (B, n, T, h)
        k = self.wk(x).view(B, T, self.n, self.h).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n, self.h).transpose(1, 2)

        q, k = self.rotary(q), self.rotary(k)
        att_val = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        attention = att_val.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.wo(attention))

class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.w1 = nn.Linear(config.d_model, config.d_model * 4)
        self.w2 = nn.Linear(config.d_model * 4, config.d_model)
        self.act = ReLUSquared()
        self.dropout = nn.Dropout(config.dropout)
        self.w2.flag = 1

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.dropout(x)
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
        x = x + self.attn(self.rmsn1(x)) 
        x = x + self.ffn(self.rmsn2(x))
        return x

class Gpt(nn.Module):
    def __init__(self, config):
        super(Gpt, self).__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.final_rms = RMSNorm(config)
        self.final = nn.Linear(config.d_model, config.vocab_size, bias = False)
        print("Number of params: ", sum(p.numel() for p in self.parameters()))
        self.final.weight.data.zero_()
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('final.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))


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


#------------------------------------Training---------------------------------------------------
model = Gpt(config).to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    model_clone = model.module
else:
    model_clone = model
model = torch.compile(model)  

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

optimizer = AdamMini(
    model,
    config.vocab_size,
    config.n_heads,
    lr=config.learning_rate,      
    weight_decay=config.weight_decay 
)
# optimizer = torch.optim.AdamW(
#     model_clone.parameters(),
#     lr=config.learning_rate,
#     weight_decay=config.weight_decay
# )

print("Compilation done")
warmup_steps = 2000
max_steps = 200000
max_lr = 8e-4
min_lr = 0.0
eval_iters = 10

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_rate = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_rate <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_rate)) 
    return min_lr + coeff * (max_lr - min_lr)

def train(model, ctx, opt, config, epochs, grad_accum_steps=16):
    log_file = f"./logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and config.dtype == 'float16'))
    x, y = get_batch('train')
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
        current_step = epoch * grad_accum_steps
        for param_group in opt.param_groups:
            param_group['lr'] = get_lr(current_step)
        model.train()
        opt.zero_grad(set_to_none=True) 
        
        cumulative_loss = torch.zeros(1, device=device)
        start_time = time.time()

        for step in range(grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (step == grad_accum_steps - 1)

            with ctx:
                y_pred, loss = model(x, y)
                x, y = get_batch('train')
                loss = loss / grad_accum_steps

            cumulative_loss += loss.detach()
            
            scaler.scale(loss).backward()

        scaler.step(opt)
        val_x, val_y = get_batch('val')
        scaler.update()

        if ddp:
            dist.all_reduce(cumulative_loss, op=dist.ReduceOp.AVG)
            torch.distributed.barrier()
        tokens_per_batch = config.batch_size * config.context_length * grad_accum_steps
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_per_batch / elapsed

        model.eval()
        with torch.no_grad():
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
        )

        torch.cuda.empty_cache()


data_dir = "./fineweb10B"
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config.context_length, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.context_length]).astype(np.int64)) for i in ix])
    if device.type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def save_model(model, model_name="adam-gpt.pt"):
    torch.save(model.state_dict(), model_name) 

try:
    train(model, ctx, optimizer, config, config.max_epochs)
    if ddp:
        destroy_process_group()
    if master:
        save_model(model_clone, model_name="adamGpt.pt")
except Exception as e:
    print(f"Training failed: {e}")
    if ddp:
        destroy_process_group()
    raise e

