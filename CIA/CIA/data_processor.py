import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from torch_geometric.data import Data
import networkx as nx
from collections import defaultdict
import os
from pathlib import Path
from difflib import SequenceMatcher

class GraphDataProcessor:
    
    def __init__(self):
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, te.
        xt1.lower(), text2.lower()).ratio()
    
    def _match_nodes_to_outputs(self, nodes: List[Dict], output_list: List[str]) -> Dict[str, int]:
        node_id_to_idx = {}
        used_indices = set()
        max_idx = len(output_list) - 1 
        for node in nodes:
            node_output = node.get('node_outputs', '')
            if not node_output:
                for idx in range(max_idx + 1):
                    if idx not in used_indices:
                        node_id_to_idx[node['node_id']] = idx
                        used_indices.add(idx)
                        break
                else:
                    max_idx += 1
                    node_id_to_idx[node['node_id']] = max_idx
                    used_indices.add(max_idx)
                continue
            
            similarities = []
            for idx, output in enumerate(output_list):
                if idx in used_indices:
                    similarity = -1 
                else:
                    similarity = self._calculate_similarity(node_output, output)
                similarities.append((idx, similarity))
            
            best_match = max(similarities, key=lambda x: x[1])
            if best_match[1] > 0:
                matched_idx = best_match[0]
                node_id_to_idx[node['node_id']] = matched_idx
                used_indices.add(matched_idx)
            else:

                for idx in range(max_idx + 1):
                    if idx not in used_indices:
                        node_id_to_idx[node['node_id']] = idx
                        used_indices.add(idx)
                        break
                else:
                    max_idx += 1
                    node_id_to_idx[node['node_id']] = max_idx
                    used_indices.add(max_idx)
        
        return node_id_to_idx
        
    def load_data(self, json_file_path: str) -> List[Dict]:
        path = Path(json_file_path)
        all_data = []
        json_files = path.glob('*.json')
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                all_data.extend(file_data)
        return all_data

    
    def build_edge_index(self, graph_data: Dict) -> torch.Tensor:
        output_list = graph_data[0] 
        nodes = graph_data[1]['nodes']
        node_id_to_idx = self._match_nodes_to_outputs(nodes, output_list)
        self.node_id_to_idx = node_id_to_idx.copy()
        self.idx_to_node_id = {idx: node_id for node_id, idx in node_id_to_idx.items()}
        edges = []
        for node in nodes:
            node_idx = self.node_id_to_idx[node['node_id']]
            
            for pred in node.get('spatial_predecessors', []):
                if '(' in pred:
                    pred_id = pred.split('(')[0]
                    if pred_id in self.node_id_to_idx:
                        pred_idx = self.node_id_to_idx[pred_id]
                        edges.append([pred_idx, node_idx])
            
            for succ in node.get('spatial_successors', []):
                if '(' in succ:
                    succ_id = succ.split('(')[0]
                    if succ_id in self.node_id_to_idx:
                        succ_idx = self.node_id_to_idx[succ_id]
                        edges.append([node_idx, succ_idx])
        
        for node in nodes:
            node_idx = self.node_id_to_idx[node['node_id']]
            
            for pred in node.get('temporal_predecessors', []):
                if '(' in pred:
                    pred_id = pred.split('(')[0]
                    if pred_id in self.node_id_to_idx:
                        pred_idx = self.node_id_to_idx[pred_id]
                        edges.append([pred_idx, node_idx])
            
            for succ in node.get('temporal_successors', []):
                if '(' in succ:
                    succ_id = succ.split('(')[0]
                    if succ_id in self.node_id_to_idx:
                        succ_idx = self.node_id_to_idx[succ_id]
                        edges.append([node_idx, succ_idx])
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index
    
    def build_adjacency_matrix(self, graph_data: Dict) -> torch.Tensor:
        nodes = graph_data[1]['nodes']
        num_nodes = len(nodes)+1
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for node in nodes:
            node_idx = self.node_id_to_idx[node['node_id']]
            for pred in node.get('spatial_predecessors', []):
                if '(' in pred:
                    pred_id = pred.split('(')[0]
                    if pred_id in self.node_id_to_idx:
                        pred_idx = self.node_id_to_idx[pred_id]
                        adj_matrix[pred_idx, node_idx] = 1

            for succ in node.get('spatial_successors', []):
                if '(' in succ:
                    succ_id = succ.split('(')[0]
                    if succ_id in self.node_id_to_idx:
                        succ_idx = self.node_id_to_idx[succ_id]
                        adj_matrix[node_idx, succ_idx] = 1

        for node in nodes:
            node_idx = self.node_id_to_idx[node['node_id']]
            for pred in node.get('temporal_predecessors', []):
                if '(' in pred:
                    pred_id = pred.split('(')[0]
                    if pred_id in self.node_id_to_idx:
                        pred_idx = self.node_id_to_idx[pred_id]
                        adj_matrix[pred_idx, node_idx] = 1

            for succ in node.get('temporal_successors', []):
                if '(' in succ:
                    succ_id = succ.split('(')[0]
                    if succ_id in self.node_id_to_idx:
                        succ_idx = self.node_id_to_idx[succ_id]
                        adj_matrix[node_idx, succ_idx] = 1

        
        return adj_matrix
    
    def extract_induction_output(self, graph_data: Dict) -> str:
        induction_output = []
        for data in graph_data:
            induction_output.append(data[0])
        return induction_output
    
    def process_single_graph(self, graph_data: Dict) -> Dict[str, Any]:
        edge_index = self.build_edge_index(graph_data[0])
        adj_matrix = self.build_adjacency_matrix(graph_data[0])
        induction_output=self.extract_induction_output(graph_data[0][0])
        return {
            'edge_index': edge_index,
            'adjacency_matrix': adj_matrix,
            'num_nodes': len(graph_data[0][1]['nodes'])+1,
            'induction_output': induction_output
        }
    
    def process_all_graphs(self, json_file_path: str) -> List[Dict[str, Any]]:
        data = self.load_data(json_file_path)
        processed_graphs = []
        for graph_data in data:
            processed_graph = self.process_single_graph(graph_data)
            processed_graphs.append(processed_graph)    
        return processed_graphs
    
    def create_pyg_data(self, processed_graph: Dict[str, Any]) -> Data:

        return Data(
            x=torch.zeros(processed_graph['num_nodes'], 1),  
            edge_index=processed_graph['edge_index'],
            y=processed_graph['adjacency_matrix']
        )

class GraphDataset:

    
    def __init__(self, processed_graphs: List[Dict[str, Any]]):
        self.processed_graphs = processed_graphs
        self.processor = GraphDataProcessor()
    
    def __len__(self):
        return len(self.processed_graphs)
    
    def __getitem__(self, idx):
        return self.processed_graphs[idx]
    
    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:

        return [self.processed_graphs[i] for i in indices]
    
    def split_train_test(self, test_ratio: float = 0.2) -> Tuple['GraphDataset', 'GraphDataset']:
        np.random.seed(42)
        indices = np.random.permutation(len(self.processed_graphs))
        split_idx = int(len(indices) * (1 - test_ratio))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_graphs = [self.processed_graphs[i] for i in train_indices]
        test_graphs = [self.processed_graphs[i] for i in test_indices]
        
        return GraphDataset(train_graphs), GraphDataset(test_graphs)

def analyze_graph_statistics(processed_graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        'num_graphs': len(processed_graphs),
        'num_nodes_per_graph': [],
        'num_edges_per_graph': [],
        'density_per_graph': [],
        'avg_degree_per_graph': []
    }
    
    for graph in processed_graphs:
        num_nodes = graph['num_nodes']
        num_edges = graph['edge_index'].size(1)
        
        stats['num_nodes_per_graph'].append(num_nodes)
        stats['num_edges_per_graph'].append(num_edges)
        

        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0
        stats['density_per_graph'].append(density)

        adj_matrix = graph['adjacency_matrix']
        degrees = adj_matrix.sum(dim=1)
        avg_degree = degrees.mean().item()
        stats['avg_degree_per_graph'].append(avg_degree)
    

    stats['summary'] = {
        'avg_num_nodes': np.mean(stats['num_nodes_per_graph']),
        'std_num_nodes': np.std(stats['num_nodes_per_graph']),
        'avg_num_edges': np.mean(stats['num_edges_per_graph']),
        'std_num_edges': np.std(stats['num_edges_per_graph']),
        'avg_density': np.mean(stats['density_per_graph']),
        'std_density': np.std(stats['density_per_graph']),
        'avg_degree': np.mean(stats['avg_degree_per_graph']),
        'std_degree': np.std(stats['avg_degree_per_graph'])
    }
    
    return stats
