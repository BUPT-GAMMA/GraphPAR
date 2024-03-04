from abc import ABC, abstractmethod

import torch


class NoiseAdder(ABC):

    def __init__(self, attribute_vector):
        self.attribute_vector = attribute_vector
        self.cur_batch_size = -1
        self.attribute_vector_repeated = None

    def set_attribute_vectors_repeated(self, required_batch_size: int):
        if self.cur_batch_size != required_batch_size:
            self.cur_batch_size = required_batch_size
            self.attribute_vector_repeated = torch.repeat_interleave(self.attribute_vector.unsqueeze(0),
                                                                     self.cur_batch_size,
                                                                     dim=0)

    def add_noise(self, z_encoder: torch.Tensor):
        self.set_attribute_vectors_repeated(z_encoder.size(0))
        return self._add_noise(z_encoder)

    @abstractmethod
    def _add_noise(self, z_gen_model_latents: torch.Tensor):
        pass


class GaussianNoiseAdder(NoiseAdder):

    def __init__(self, attribute_vector: torch.Tensor, sigma: float):
        super(GaussianNoiseAdder, self).__init__(attribute_vector)
        self.sigma = sigma

    def _add_noise(self, z_gen_model_latents: torch.Tensor):
        noisy_latents = z_gen_model_latents.clone().detach()
        coeffs = torch.randn(self.cur_batch_size, 1, device=noisy_latents.device) * self.sigma
        noisy_latents += self.attribute_vector_repeated * coeffs
        return noisy_latents
