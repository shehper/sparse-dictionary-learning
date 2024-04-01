import torch
import torch.nn.functional as F
from model import GPT

class HookedGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.mlp_activation_hooks = []
        self.hook_handle = None

    def hook_fn(self, module, input, output, mode='store', replacement_tensor=None):
        if mode == 'store':
            self.mlp_activation_hooks.append(output.clone().detach())
        elif mode == 'replace':
            if replacement_tensor is None:
                replacement_tensor = torch.zeros_like(output)
            return replacement_tensor

    def forward(self, idx, targets=None, mode='store', replacement_tensor=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        if mode == 'store' and self.mlp_activation_hooks:
            self.clear_mlp_activation_hooks()

        # Register the hook on the MLP GELU activation of the last transformer block
        self.hook_handle = self.transformer.h[-1].mlp.gelu.register_forward_hook(
            lambda module, input, output: 
            self.hook_fn(module, input, output, mode=mode, replacement_tensor=replacement_tensor)
        )

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        # Remove the hook after the forward pass
        self.hook_handle.remove()
        self.hook_handle = None

        return logits, loss

    def clear_mlp_activation_hooks(self):
        self.mlp_activation_hooks.clear()