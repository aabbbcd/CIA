import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Any, Set
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch_geometric.nn import GATConv, global_mean_pool
import json
import os
import pickle
import re
from utils import SemanticGraphBuilder, GraphEncoder, SentenceTransformerEncoder, Encoder, Decoder, TCLineEstimator, TCTreeEstimator
from utils import _client
import random

class PerGraphSharedGenerator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, graph_embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(graph_embeddings)

class SelfSupervisedModel(nn.Module):
    
    def __init__(self, text_encoder_name: str = "/data/llm/all-MiniLM-L6-v2", device: Optional[torch.device] = None, hidden_dim: int = 256,  dropout: float = 0.1, tokenizer_max_length: int =2048, data_path: str = None, encoder: str = None, max_agents: int = 5):
        super().__init__()
        self.device = device

        self.encoder = encoder
        self.sentence_encoder = SentenceTransformerEncoder(device=self.device)
        self.sentence_dim = self.sentence_encoder.embedding_dim
        
        self.shared_gen = PerGraphSharedGenerator(dim=self.sentence_dim)
        input_dim = self.sentence_dim

        self.mi_calc = TCLineEstimator(hidden_size=hidden_dim, mi_estimator="InfoNCE", device=self.device).to(self.device)
        self.encoder_p = Encoder(input_dim, self.device)
        self.encoder_s = Encoder(input_dim, self.device)
        self.fusion = nn.Linear(input_dim * 4, input_dim*2).to(self.device)
        self.decoder = Decoder(input_dim, device=self.device)
        self.link_predictor = nn.Bilinear(input_dim*2, input_dim*2, 1) 
       
        self.ranking_loss = nn.MarginRankingLoss(margin=0.5)
        self.sup_criterion = nn.BCEWithLogitsLoss()
    def get_sentence_embeddings(self, node_outputs: List[str]) -> List[torch.Tensor]:
        client = _client
        SYSTEM = (
            "You are a code analyzer. Your task is to analyze code snippets and explain what they do based on the context and task description.\n"
            "CRITICAL RULES:\n"
            "1. Return ONLY a plain text explanation\n"
            "2. No JSON, no markdown, no code blocks, no extra formatting\n"
            "3. Write a clear, concise description of what the code does\n"
            "4. Focus on functionality and purpose\n"
            "5. Explain the main operations and their purpose\n"
            "7. If no code is present, describe the general content\n\n"
            "Example: \"This code reads data from a file, processes it through calculations, and outputs the results to another file.\""
        )
        USER_TPL = (
            "Analyze the following text and explain what the code does based on the task description and code context.\n\n"
            "Text:\n<<<\n{text}\n>>>\n\n"
            "Provide a clear explanation of what this code accomplishes:"
        )


        for i in range(len(node_outputs)):
            for j in range(len(node_outputs[i])):
                if "def" in node_outputs[i][j]:
                    messages = [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER_TPL.format(text=node_outputs[i][j])}
                    ]
                    response = client.chat.completions.create(
                        model="gemini-2.5-pro",
                        messages=messages,
                    )
                    node_outputs[i][j] = response.choices[0].message.content


        g_list = []
        for i, text in enumerate(node_outputs):
            g_i = self.sentence_encoder.encode_texts(text)
            g_list.append(g_i)
            
        return g_list

    def forward(self, node_outputs: List[str], pair_log, training: bool = True) -> Dict[str, torch.Tensor]:
        g_list = self.get_sentence_embeddings(node_outputs)
        g_stack = torch.stack(g_list, dim=0)
        if g_stack.dim() == 3 and g_stack.size(1) == 1:
            g_stack = g_stack.squeeze(1)

        p_stack = self.encoder_p(g_stack)
        s_stack = self.encoder_s(g_stack)

        res = {
            "s_loss": torch.tensor(0.0, device=self.device),
            "p_loss": torch.tensor(0.0, device=self.device),
            "rec_loss": torch.tensor(0.0, device=self.device),
            "align_loss": torch.tensor(0.0, device=self.device),
            "var_p": torch.tensor(0.0, device=self.device),
            "sup_loss": torch.tensor(0.0, device=self.device)
        }
        self.p_list = list(p_stack.unbind(0)) 
        self.s_list = list(s_stack.unbind(0))
        if training:
            combined_features = torch.cat([p_stack, s_stack], dim=-1)
            z_fused = F.relu(self.fusion(combined_features))
            rec_g = self.decoder(z_fused)
            rec_loss = F.mse_loss(rec_g, g_stack)
            s_list_2d = [s.unsqueeze(0) if s.dim() == 1 else s for s in self.s_list]
            s_loss = self.mi_calc(s_list_2d)
            p_list_2d = [p.unsqueeze(0) if p.dim() == 1 else p for p in self.p_list]
            align_loss_list = []
            for i in range(len(p_list_2d)):
                align_loss_list.append(self.mi_calc([p_list_2d[i], s_list_2d[i]]))
            align_loss = sum(align_loss_list)
            feature_var = p_stack.var(dim=0, unbiased=False)
            target_var = 1.0
            var_p = F.relu(target_var - feature_var).mean()
            sup_loss = torch.tensor(0.0, device=self.device)
            if pair_log is not None and len(pair_log) > 0:
                num_nodes = len(node_outputs)
                pos_edge_set = set(tuple(e) for e in pair_log)
                all_possible_edges = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            all_possible_edges.append((i, j))
                
                pos_indices = [e for e in all_possible_edges if e in pos_edge_set]
                unlabeled_indices = [e for e in all_possible_edges if e not in pos_edge_set]
                if len(pos_indices) > 0 and len(unlabeled_indices) > 0:
                    p_src_pos = F.normalize(torch.stack([self.p_list[i] for i, j in pos_indices]), p=2, dim=1)
                    p_dst_pos = F.normalize(torch.stack([self.p_list[j] for i, j in pos_indices]), p=2, dim=1)
                    if len(unlabeled_indices) >= len(pos_indices):
                        sampled_unlabeled = random.sample(unlabeled_indices, len(pos_indices))
                    else:
                        sampled_unlabeled = random.choices(unlabeled_indices, k=len(pos_indices))                
                    p_src_un = F.normalize(torch.stack([self.p_list[i] for i, j in sampled_unlabeled]), p=2, dim=1)
                    p_dst_un = F.normalize(torch.stack([self.p_list[j] for i, j in sampled_unlabeled]), p=2, dim=1)
                    scores_pos = self.link_predictor(p_src_pos, p_dst_pos)
                    scores_un = self.link_predictor(p_src_un, p_dst_un)                    
                    all_scores = torch.cat([scores_pos, scores_un], dim=0)
                    labels_pos = torch.full_like(scores_pos, 0.8) 
                    labels_un = torch.full_like(scores_un, 0.2)
                    all_labels = torch.cat([labels_pos, labels_un], dim=0)                    
                    sup_loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
            res["s_loss"] = s_loss
            res["rec_loss"] = rec_loss
            res["align_loss"] = align_loss
            res["sup_loss"] = sup_loss
        
        with torch.no_grad():
            sim = torch.zeros(len(node_outputs), len(node_outputs), device=self.device)
            for i in range(len(node_outputs)):
                for j in range(i, len(node_outputs)):
                    if i == j:
                        sim[i][j] = 0
                    else:
                        p_i_norm = F.normalize(self.p_list[i].unsqueeze(0), p=2, dim=1)
                        p_j_norm = F.normalize(self.p_list[j].unsqueeze(0), p=2, dim=1)
                        out_ij = self.link_predictor(p_i_norm, p_j_norm).mean(dim=1).squeeze()
                        out_ji = self.link_predictor(p_j_norm, p_i_norm).mean(dim=1).squeeze()
                        sim_value = (out_ij + out_ji) / 2.0
                        sim[i][j] = sim_value
                        sim[j][i] = sim_value
            mask = ~torch.eye(len(node_outputs), dtype=torch.bool, device=sim.device)
            non_diag_values = sim[mask]
            if len(non_diag_values) > 0:
                sim_min = non_diag_values.min()
                sim_max = non_diag_values.max()
                if sim_max > sim_min: 
                    sim_normalized = (sim - sim_min) / (sim_max - sim_min)
                    sim = sim_normalized * mask.float()
            res["sim"] = sim        
        return res

    @staticmethod
    def loss_self_supervised(
        s_loss,
        rec_loss,   
        align_loss,
        sup_loss,
        sim: torch.Tensor,
    ) -> torch.Tensor:

        return rec_loss - s_loss+ align_loss + 0.1*sup_loss 
