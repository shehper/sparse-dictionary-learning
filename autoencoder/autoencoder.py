import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class AutoEncoder(nn.Module):
    def __init__(self, n, m, lam=0.003, resampling_interval=None):
        # for us, n = d_MLP (a.k.a. n_ffwd), m = number of autoencoder neurons
        super().__init__()
        self.n, self.m = n, m
        self.enc = nn.Linear(n, m)
        self.relu = nn.ReLU()
        self.dec = nn.Linear(m, n)
        self.lam = lam # coefficient of L_1 loss

        # some variables that are needed if resampling neurons
        self.resampling_interval = resampling_interval
        self.dead_neurons = None

        # normalize dictionary vectors to have unit norm
        self.normalize_decoder_columns() 

    def forward(self, x):
        # x is of shape (b, n) where b = batch_size, n = d_MLP

        xbar = x - self.dec.bias # (b, n)
        f = self.relu(self.enc(xbar)) # (b, m)
        reconst_acts = self.dec(f) # (b, n)
        mseloss = F.mse_loss(reconst_acts, x) # scalar
        l1loss = F.l1_loss(f, torch.zeros(f.shape, device=f.device), reduction='sum') # scalar
        loss = mseloss + self.lam * l1loss # scalar
        
        # if in training phase (i.e. model.train() has been called), we only need f and loss
        # but if evaluating (i.e. model.eval() has been called), we will need reconstructed activations and other losses as well
        out_dict = {'loss': loss, 'f': f} if self.training else {'loss': loss, 'f': f, 'reconst_acts': reconst_acts, 'mse_loss': mseloss, 'l1_loss': l1loss}
        
        return out_dict

    @torch.no_grad()
    def normalize_decoder_columns(self):
        self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)

    def remove_parallel_component_of_decoder_gradient(self):
        # remove gradient information parallel to weight vectors
        # to do so, compute projection of gradient onto weight
        # recall projection of a onto b is proj_b a = (a.\hat{b}) \hat{b}
        unit_w = F.normalize(self.dec.weight, dim=0) # \hat{b}
        proj = torch.sum(self.dec.weight.grad * unit_w, dim=0) * unit_w 
        self.dec.weight.grad = self.dec.weight.grad - proj

    @torch.no_grad()
    def initiate_dead_neurons(self):
        self.dead_neurons = set([neuron for neuron in range(self.m)])

    @torch.no_grad()
    def update_dead_neurons(self, f):
        # obtain indices to columns of f (i.e. neurons) that fire on at least one example
        active_neurons_this_step = torch.count_nonzero(f, dim=0).nonzero().view(-1)
        
        # remove these neurons from self.dead_neurons
        for neuron in active_neurons_this_step:
            self.dead_neurons.discard(neuron.item())

    @torch.no_grad()
    def resample_neurons(self, data, optimizer, batch_size=8192):

        if not self.dead_neurons:
            return

        device = next(self.parameters()).device # if all model parameters are on the same device (which in our case is True), use this to get that device
        dead_neurons_t = torch.tensor(list(self.dead_neurons))
        alive_neurons = torch.tensor([i for i in range(self.m) if i not in self.dead_neurons])

        # compute average norm of encoder vectors for alive neurons
        average_enc_norm = torch.mean(torch.linalg.vector_norm(self.enc.weight[alive_neurons], dim=1))
        
        # compute probs = loss^2 for all examples in data
        # expect data to be of shape (N, n_ffwd); in the paper N = 819200
        num_batches = len(data) // batch_size + (len(data) % batch_size != 0)
        probs = torch.zeros(len(data),) # (N, ) # initiate a tensor of probs = losses**2
        for iter in range(num_batches): 
            print(f'computing probs=losses**2 for iter = {iter}/{num_batches} for neuron resampling')
            x = data[iter * batch_size: (iter + 1) * batch_size].to(device) # (b, n) where b = min(batch_size, remaining examples in data), n = d_MLP
            xbar = x - self.dec.bias # (b, n)
            f = self.relu(self.enc(xbar)) # (b, m)
            reconst_acts = self.dec(f) # (b, n)
            mselosses = torch.sum(F.mse_loss(reconst_acts, x, reduction='none'), dim=1) # (b,)
            l1losses = torch.sum(F.l1_loss(f, torch.zeros(f.shape, device=f.device), reduction='none'), dim=1) # (b, )
            probs[iter * batch_size: (iter + 1) * batch_size] = ((mselosses + self.lam * l1losses)**2).to('cpu') # (b, )

        # pick examples based on probs
        exs = data[torch.multinomial(probs, num_samples=len(self.dead_neurons))].to(dtype=torch.float32) # (d, n) where d = len(dead_neurons)
        assert exs.shape == (len(self.dead_neurons), self.n), 'exs has incorrect shape'
        
        # normalize examples to have unit norm and reset decoder weights for dead neurons
        exs_unit_norm = F.normalize(exs, dim=1).to(device) # (d, n)
        self.dec.weight[:, dead_neurons_t] = torch.transpose(exs_unit_norm, 0, 1) # (n, d)
        
        # renormalize examples to have norm = average_enc_norm * 0.2 and reset encoder weights and biases
        exs_enc_norm = exs_unit_norm * average_enc_norm * 0.2
        self.enc.weight[dead_neurons_t] = exs_enc_norm
        self.enc.bias[dead_neurons_t] = 0

        # update Adam parameters associated to decoder weights, and encoder weights and bias
        for i, p in enumerate(optimizer.param_groups[0]['params']): # there is only one parameter group so we can do this
            param_state = optimizer.state[p]
            if i in [0, 1]: # encoder weight and bias
                param_state['exp_avg'][dead_neurons_t] = 0
                param_state['exp_avg_sq'][dead_neurons_t] = 0
            elif i == 2: # decoder weight
                param_state['exp_avg'][:, dead_neurons_t] = 0
                param_state['exp_avg_sq'][:, dead_neurons_t] = 0

        print('Dead neurons resampled successfully!')

        # reset self.dead_neurons as there are now none left to be resampled
        self.dead_neurons = None