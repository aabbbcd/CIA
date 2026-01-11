import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from torch_geometric.data import Data
import networkx as nx
from collections import defaultdict

class GraphDataProcessor:
    
    def __init__(self):
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        
    def load_data(self, json_file_path: str) -> List[Dict]:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    
    def build_edge_index(self, graph_data: Dict) -> torch.Tensor:
        nodes = graph_data['nodes']
        self.node_id_to_idx = {node['node_id']: idx for idx, node in enumerate(nodes,start=1)}
        self.idx_to_node_id = {idx: node['node_id'] for idx, node in enumerate(nodes,start=1)}
        self.node_id_to_idx[graph_data["decision"]['decision_node_id']]=0
        self.idx_to_node_id[0]=graph_data["decision"]['decision_node_id']
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
        num_nodes = len(graph_data['nodes'])+1
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for node in graph_data['nodes']:
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

        for node in graph_data['nodes']:
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
    
    def process_single_graph(self, graph_data: Dict) -> Dict[str, Any]:
        edge_index = self.build_edge_index(graph_data[0])
        adj_matrix = self.build_adjacency_matrix(graph_data[0])
        task=[]
        for i in range(1, len(graph_data)):
            task.append(graph_data[i]['task'])
        return {
            'edge_index': edge_index,
            'adjacency_matrix': adj_matrix,
            'num_nodes': len(graph_data[0]['nodes'])+1,
            'graph_id': graph_data[0].get('graph_id', 'unknown'),
            'task':task
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
