import torch
import torch.nn as nn
import numpy as np
import math
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from torchvision import models
from typing import Optional, List, Tuple
import os
import json
import re
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
_client = OpenAI(base_url="",api_key="")


class SentenceTransformerEncoder(nn.Module):
	def __init__(self, device, model_name: str = "/data/llm/all-MiniLM-L6-v2"):
		super().__init__()
		self.device = device
		self.model = SentenceTransformer(model_name, device=self.device)
		self.embedding_dim = self.model.get_sentence_embedding_dimension()
		
	@torch.no_grad()
	def encode_texts(self, texts: List[str]) -> torch.Tensor:
		embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
		if len(texts) == 1:
			if embeddings.dim() == 1:
				embeddings = embeddings.unsqueeze(0)
		embeddings = embeddings.to(self.device)
		return embeddings

class Encoder(nn.Module):
	def __init__(self, z_dim, device: Optional[torch.device] = None):
		super(Encoder, self).__init__()
		self.device = device
		self.z_dim = z_dim
		self.in_dim = z_dim
		# Vanilla MLP
		self.net = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(self.in_dim, 1024),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(1024, self.z_dim * 2),
		)
		self.net = self.net.to(self.device)
	def forward(self, x):

		params = self.net(x).to(self.device)
		return params


class Decoder(nn.Module):
	def __init__(self,z_dim, device: Optional[torch.device] = None, scale=0.39894):
		super(Decoder, self).__init__()
		self.device = device
		self.z_dim = z_dim
		self.scale = scale
		self.out_dim =  z_dim
		# Vanilla MLP
		self.net = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(self.z_dim*2, 1024),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(1024, self.out_dim)
		)
		self.net = self.net.to(self.device)

	def forward(self, z):
		x = self.net(z).to(self.device)
		return x



	
class InfoNCE(nn.Module):
	def __init__(self, x_dim, y_dim, hidden_size=None):
		super(InfoNCE, self).__init__()

		if hidden_size is None:
			self.F_func = nn.Linear(x_dim + y_dim, 1)
		else:
			self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, int(hidden_size)),
									nn.ReLU(),
									nn.Linear(int(hidden_size), 1))
	
	def forward(self, x_samples, y_samples):  

		sample_size = y_samples.shape[0]

		x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
		y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

		T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
		T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

		lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
		return lower_bound

	def learning_loss(self, x_samples, y_samples):
		return -self.forward(x_samples, y_samples)




MI_CLASS={
	'InfoNCE': InfoNCE,
	}

class TCLineEstimator(nn.Module):
	def __init__(self, hidden_size=None, mi_estimator='InfoNCE',device=None):  
		super().__init__()
		self.hidden_size = hidden_size
		self.mi_estimator = mi_estimator
		self.mi_est_type = MI_CLASS[mi_estimator]
		self.device = device
		
	
	def forward(self, samples): 
		samples = [torch.as_tensor(s) if not isinstance(s, torch.Tensor) else s for s in samples]
		dims = [samples[i].shape[1] for i in range(len(samples))]
		mi_estimator_list = [
			self.mi_est_type(
				x_dim=sum(dims[:i+1]),
				y_dim=dim,
				hidden_size=(None if self.hidden_size is None else self.hidden_size * np.sqrt(i+1))
			).to(self.device)
			for i, dim in enumerate(dims[:-1])
		]
			
		self.mi_estimators = nn.ModuleList(mi_estimator_list)
		
		outputs = []
		concat_samples = [samples[0]]
		for i, dim in enumerate(dims[1:]):
			cat_sample = torch.cat(concat_samples, dim=1)
			outputs.append(self.mi_estimators[i](cat_sample, samples[i+1]))
			concat_samples.append(samples[i+1])
		return torch.stack(outputs).sum()

	def learning_loss(self, samples):
		samples = [torch.as_tensor(s) if not isinstance(s, torch.Tensor) else s for s in samples]
		dims = [samples[i].shape[1] for i in range(len(samples))]
		mi_estimator_list = [
			self.mi_est_type(
				x_dim=sum(dims[:i+1]),
				y_dim=dim,
				hidden_size=(None if self.hidden_size is None else self.hidden_size * np.sqrt(i+1))
			).to(self.device)
			for i, dim in enumerate(dims[:-1])
		]
			
		self.mi_estimators = nn.ModuleList(mi_estimator_list)
		
		outputs = []
		concat_samples = [samples[0]]
		for i, dim in enumerate(dims[1:]):
			cat_sample = torch.cat(concat_samples, dim=1)
			outputs.append(self.mi_estimators[i].learning_loss(cat_sample, samples[i+1]))
			concat_samples.append(samples[i+1])

		return torch.stack(outputs).mean()

