import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import *
import math
import inspect

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        """
        Weight Tying improves the performance of language models by tying (sharing) the weights of the embedding and softmax layers.
        This method also massively reduces the total number of parameters in the language models that it is applied to. 
        """
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # Init all weights
        self.apply(self._init_weight)

        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print("Number of parameters: %.2fM" % (self.get_num_params()%1e6))
    
    def get_num_params(self, non_embedding=True):

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:,[-1],:])
            loss = None
        
        return logits, loss

    def crop_block_size(self, block_size):
        #Model surgery to decrease the block size if necessary
        # we may load GPT2 pretrained model checkpoint

        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    
    def configure_optimizers(self, weight_decay, lr_rate, betas, device_type):
        """
        Configure Optimizers and its parameters.
        """
        
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        #filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}


        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay':weight_decay},
            {'params': nodecay_params, 'weight_decay':0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr_rate, betas=betas, **extra_args)
        print(f"Using Fused AdamW: {use_fused}")

        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        fwdbwd_per_iter = indicates how many forward and backward propagations are performed in each iteration
        dt = Specifies the duration of an iteration in seconds.

        Purpose:
            Calculate Model Flops Utilization (MFU)
            MFU: Shows the data of the model on the hardware.

        """

        N = self.get_num_params()
        L, H, Q, T = config.n_layer, config.n_head, config.n_embd//config.n_head, config.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        
        #MFU ---> Model Flops Utilization
        return mfu
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_token times, feeding the predictions back into the model each time.
        Most likely you will wanna make sure to be in model.eval() mode of operation for this.
        """

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logist, _ = self(idx_cond)
            #forward the model to get the logits for the index in the sequence
            logist = logist[:, -1, :] / temperature
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logist, min(top_k, logist.size(-1)))
                logist[logist < v[:, -1]] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logist, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == '__main__':
    try:
        config = GPTConfig()
        model = GPT(config)

        print(f'Number of parameters: {model.get_num_params() / 1e6: .2f}M')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        input_ids = torch.randint(0, config.vocab_size, (1,10)).to(device)
        logits, loss = model(input_ids)
        print(f'Logits shape: {logits.shape}')
        print(f"Loss: {loss}")
        generated_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.7, top_k=40)
        def decode(idx):
            return " ".join([f'Token_{i}' for i in idx.tolist()])
        print('Generated text:\n ',decode(generated_ids[0]))
    except Exception as e:
        print('Error message: {}'.format(e))