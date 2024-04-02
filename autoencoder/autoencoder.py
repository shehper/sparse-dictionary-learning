"""
This file defines an AutoEncoder class, which also contains an implementation of neuron resampling.   
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class AutoEncoder(nn.Module):
    def __init__(self, n_inputs: int, n_latents: int, lam: float = 0.003, resampling_interval: int = 25000):
        """
        n_input: Number of inputs
        n_latents: Number of neurons in the hidden layer
        lam: L1-coefficient for Sparse Autoencoder
        resampling_interval: Number of training steps after which dead neurons will be resampled
        """
        super().__init__()
        self.n_inputs, self.n_latents = n_inputs, n_latents
        self.encoder = nn.Linear(n_inputs, n_latents)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(n_latents, n_inputs)
        self.lam = lam
        self.resampling_interval = resampling_interval
        self.dead_neurons = None
        self.normalize_decoder_columns()

    def forward(self, x):
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        loss = self.calculate_loss(x, latents, reconstructed)

        if self.training:
            return {'loss': loss, 'latents': latents}
        else:
            return {
                'loss': loss,
                'latents': latents,
                'reconst_acts': reconstructed,
                'mse_loss': self.mse_loss(reconstructed, x),
                'l1_loss': self.l1_loss(latents)
            }

    def encode(self, x):
        bias_corrected_input = x - self.decoder.bias
        return self.relu(self.encoder(bias_corrected_input))

    def decode(self, encoded):
        return self.decoder(encoded)

    def calculate_loss(self, x, encoded, reconstructed):
        mse_loss = F.mse_loss(reconstructed, x)
        l1_loss = F.l1_loss(encoded, torch.zeros_like(encoded), reduction='sum') / encoded.shape[0]
        return mse_loss + self.lam * l1_loss

    @torch.no_grad()
    def get_feature_activations(self, inputs, start_idx, end_idx):
        """
        Computes the activations of a subset of features in the hidden layer.

        :param inputs: Input tensor of shape (..., n) where n = d_MLP. It includes batch dimensions.
        :param start_idx: Starting index (inclusive) of the feature subset.
        :param end_idx: Ending index (exclusive) of the feature subset.
        
        Returns the activations for the specified feature range, reducing computation by 
        only processing the necessary part of the network's weights and biases.
        """
        adjusted_inputs = inputs - self.decoder.bias  # Adjust input to account for decoder bias
        weight_subset = self.encoder.weight[start_idx:end_idx, :].t()  # Transpose the subset of weights
        bias_subset = self.encoder.bias[start_idx:end_idx]
        
        activations = self.relu(adjusted_inputs @ weight_subset + bias_subset)
        
        return activations

    @torch.no_grad()
    def normalize_decoder_columns(self):
        """
        Normalize the decoder's weight vectors to have unit norm along the feature dimension.
        This normalization can help in maintaining the stability of the network's weights.
        """
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def remove_parallel_component_of_decoder_grad(self):
        """
        Remove the component of the gradient parallel to the decoder's weight vectors.
        """
        unit_weights = F.normalize(self.decoder.weight, dim=0) # \hat{b}
        proj = (self.decoder.weight.grad * unit_weights).sum(dim=0) * unit_weights 
        self.decoder.weight.grad = self.decoder.weight.grad - proj

    @staticmethod    
    def is_dead_neuron_investigation_step(step, resampling_interval, num_resamples):
        """
        Determine if the current step is the start of a phase for investigating dead neurons.
        According to Anthropic's specified policy, it occurs at odd multiples of half the resampling interval.
        """
        return (step > 0) and step % (resampling_interval // 2) == 0 and (step // (resampling_interval // 2)) % 2 != 0 and step < resampling_interval * num_resamples

    @staticmethod
    def is_within_neuron_investigation_phase(step, resampling_interval, num_resamples):
        """
        Check if the current step is within a phase where active neurons are investigated.
        This phase occurs in intervals defined in the specified range, starting at odd multiples of half the resampling interval.
        """
        return any(milestone - resampling_interval // 2 <= step < milestone 
                   for milestone in range(resampling_interval, resampling_interval * (num_resamples + 1), resampling_interval))

    @torch.no_grad()
    def initiate_dead_neurons(self):
        self.dead_neurons = set(range(self.n_latents))

    @torch.no_grad()
    def update_dead_neurons(self, latents):
        """
        Update the set of dead neurons based on the current feature activations.
        If a neuron is active (has non-zero activation), it is removed from the dead neuron set.
        """
        active_neurons = torch.nonzero(torch.count_nonzero(latents, dim=0), as_tuple=False).view(-1)
        self.dead_neurons.difference_update(active_neurons.tolist())

    @torch.no_grad()
    def resample_dead_neurons(self, data, optimizer, batch_size=8192):
        """
        Resample the dead neurons by resetting their weights and biases based on the characteristics
        of active neurons. Proceeds only if there are dead neurons to resample.
        """
        if not self.dead_neurons:
            return

        device = self._get_device()
        dead_neurons_t, alive_neurons = self._get_neuron_indices()
        average_enc_norm = self._compute_average_norm_of_alive_neurons(alive_neurons)
        probs = self._compute_loss_probabilities(data, batch_size, device)
        selected_examples = self._select_examples_based_on_probabilities(data, probs)
        
        self._resample_neurons(selected_examples, dead_neurons_t, average_enc_norm, device)
        self._update_optimizer_parameters(optimizer, dead_neurons_t)

        print('Dead neurons resampled successfully!')
        self.dead_neurons = None

    def _get_device(self):
        return next(self.parameters()).device

    def _get_neuron_indices(self):
        dead_neurons_t = torch.tensor(list(self.dead_neurons), device=self._get_device())
        alive_neurons = torch.tensor([i for i in range(self.n_latents) if i not in self.dead_neurons], device=self._get_device())
        return dead_neurons_t, alive_neurons

    def _compute_average_norm_of_alive_neurons(self, alive_neurons):
        return torch.linalg.vector_norm(self.encoder.weight[alive_neurons], dim=1).mean()

    def _compute_loss_probabilities(self, data, batch_size, device):
        num_batches = (len(data) + batch_size - 1) // batch_size
        probs = torch.zeros(len(data), device=device)
        for i in range(num_batches):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            x_batch = data[batch_slice].to(device)
            probs[batch_slice] = self._compute_batch_loss_squared(x_batch)
        return probs.cpu()

    def _compute_batch_loss_squared(self, x_batch):
        latents = self.encode(x_batch)
        reconst_acts = self.decode(latents)
        mselosses = F.mse_loss(reconst_acts, x_batch, reduction='none').sum(dim=1)
        l1losses = F.l1_loss(latents, torch.zeros_like(latents), reduction='none').sum(dim=1)
        return (mselosses + self.lam * l1losses).square()

    def _select_examples_based_on_probabilities(self, data, probs):
        selection_indices = torch.multinomial(probs, num_samples=len(self.dead_neurons))
        return data[selection_indices].to(dtype=torch.float32)

    def _resample_neurons(self, examples, dead_neurons_t, average_enc_norm, device):
        examples_unit_norm = F.normalize(examples, dim=1).to(device)
        self.decoder.weight[:, dead_neurons_t] = examples_unit_norm.T

        # Renormalize examples to have a certain norm and reset encoder weights and biases
        adjusted_examples = examples_unit_norm * average_enc_norm * 0.2
        self.encoder.weight[dead_neurons_t] = adjusted_examples
        self.encoder.bias[dead_neurons_t] = 0

    def _update_optimizer_parameters(self, optimizer, dead_neurons_t):
        for i, param in enumerate(optimizer.param_groups[0]['params']):
            param_state = optimizer.state[param]
            if i in [0, 1]:  # Encoder weights and biases
                param_state['exp_avg'][dead_neurons_t] = 0
                param_state['exp_avg_sq'][dead_neurons_t] = 0
            elif i == 2:  # Decoder weights
                param_state['exp_avg'][:, dead_neurons_t] = 0
                param_state['exp_avg_sq'][:, dead_neurons_t] = 0
