import torch
import torch.distributed as dist
from torch.optim import Optimizer

# param_block = {}
# for name, param in model.named_parameters():
#     if 'emb' in name or ('output' in name and not 'output_rms' in name):
#         param_block[name] = {}
#         for i in range(config.vocab_size):
#             param_block[name][i] = param[i,:] if 'weight' in name else param[i]
#     elif 'query' in name or 'key' in name:
#         param_block[name] = {}
#         for i in range(config.n_heads):
#             param_block[name][i] = param[i]
#     elif 'value' in name or 'proj' in name or 'ffn' in name:
#         param_block[name] = {}
#         for i in range(param.shape[-1]):
#             param_block[name][i] = param[:, i] if 'weight' in name else param[i]
#     else:
#         param_block[name] = param


class AdamMini(Optimizer):

    def __init__(self, model, vocab_size, n_heads, lr=1e-3, beta1=0.9, beta2=0.999,
                 weight_decay=0.0, eps=1e-8):
        """
        args:
            model (torch.nn.Module): The model to optimize.
            vocab_size (int): Vocabulary size parameter.
            n_heads (int): Number of heads in the model.
            lr (float): Learning rate.
            beta1 (float): Coefficient used for computing running averages of gradient.
            beta2 (float): Coefficient used for computing running averages of squared gradient.
            weight_decay (float): Weight decay (L2 penalty).
            eps (float): Term added to the denominator to improve numerical stability.
        """
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