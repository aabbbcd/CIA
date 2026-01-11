import sys
import os
import json
import argparse
import torch
import asyncio
import copy
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Literal
import random
from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.utils.const import GDesigner_ROOT
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from datasets.mmlu_dataset import MMLUDataset


class ReasoningOutputInduction:
    
    def __init__(
        self,
        model_path: str,
        domain: str,
        llm_name: str = "gpt-5",
        device: str = "cuda:0",
        mode: str = "FullConnected",
        agent_names: List[str] = None,  
        dataset: List[Dict[str, Any]] = None,
        batch_size: int = 4,
        agent_nums: List[int] = None,
        decision_method: str = "FinalRefer",
        current_time: str = None,
        model_name: str = "GPT-5",
        num_rounds: int = 1,
    ):
        self.model_path = model_path
        self.domain = domain
        self.llm_name = llm_name
        self.device = device
        self.mode = mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.decision_method = decision_method
        self.current_time = current_time
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.agent_names = [name for name, num in zip(agent_names, agent_nums) for _ in range(num)]
        def dataloader(data_list, batch_size, i_batch):
            return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]
        kwargs = self._get_kwargs(mode, len(self.agent_names))
        graph = Graph(
            domain=domain,
            llm_name=llm_name,
            agent_names=self.agent_names,
            decision_method=decision_method,
            current_time=current_time,
            model_name=model_name,
            **kwargs
        )
        num_batches = int(len(self.dataset)/self.batch_size)
        checkpoint = torch.load(model_path, map_location='cpu')
        graph.gcn.load_state_dict(checkpoint['gcn_state_dict'])
        if checkpoint.get('mlp_state_dict') is not None and hasattr(graph, 'mlp'):
            graph.mlp.load_state_dict(checkpoint['mlp_state_dict'])
        
        
        count=0
        for i_batch in range(num_batches):
            current_batch = dataloader(self.dataset, self.batch_size, i_batch)
            if current_batch is None:
                print("No more data available.")
                break
            
            for i_record, record in enumerate(current_batch):
                count += 1
                for i in range(5):
                    R_stars = []
                    flag = True
                    while flag:
                        realized_graph = copy.deepcopy(graph)
                        realized_graph.gcn = graph.gcn
                        realized_graph.mlp = graph.mlp
                        task = record["task"]
                        answer = record["answer"]

                        input_dict = {"task": task}
                        final_answers, log_probs, decision_outputs = await realized_graph.eval_arun(input_dict, self.num_rounds)
                        
                        R_star = self.process_decision_outputs(decision_outputs)
                        if len(R_star) > 0:
                            flag = False
                    R_stars.append(R_star)
                    with open(f"/reasoning_outputs/R_stars_{count}.json", "w") as f:
                        json.dump(R_stars, f)

    
    def _get_kwargs(self, mode: str, N: int) -> Dict[str, Any]:
        initial_spatial_probability: float = 0.5
        fixed_spatial_masks: List[List[int]] = None
        initial_temporal_probability: float = 0.5
        fixed_temporal_masks: List[List[int]] = None
        node_kwargs = None
        
        def generate_layered_graph(N, layer_num=2):
            adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
            base_size = N // layer_num
            remainder = N % layer_num
            layers = []
            for i in range(layer_num):
                size = base_size + (1 if i < remainder else 0)
                layers.extend([i] * size)
            random.shuffle(layers)
            for i in range(N):
                current_layer = layers[i]
                for j in range(N):
                    if layers[j] == current_layer + 1:
                        adj_matrix[i][j] = 1
            return adj_matrix
        
        def generate_star_graph(n):
            matrix = [[0] * n for _ in range(n)]
            for i in range(0, n):
                for j in range(i+1, n):
                    matrix[i][j] = 1
            return matrix
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
        return {
            "initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs": node_kwargs
        }
    
    def process_decision_outputs(self, decision_outputs: List[str]) -> List[str]:
        R_star = []
        if not isinstance(decision_outputs, str):
            return R_star
        output = decision_outputs 
        history_match = re.search(r'\[PREVIOUS HISTORY\](.*?)\[/PREVIOUS HISTORY\]', output, re.DOTALL)
        if not history_match:
            history_match = re.search(r'\[HISTORY\](.*?)\[/HISTORY\]', output, re.DOTALL)
        if not history_match:
            return R_star
        history_content = history_match.group(1).strip()
        partitioned_results = [item.strip() for item in history_content.split('|||') if item.strip()]
        seen = set()
        deduplicated_results = []
        for item in reversed(partitioned_results):
            if item not in seen:
                seen.add(item)
                deduplicated_results.insert(0, item) 
        deduplicated_results = [item for item in deduplicated_results if item.strip().upper() != "START"]
        reasoning_match = re.search(r'\[REASONING OUTPUT\](.*?)\[/REASONING OUTPUT\]', output, re.DOTALL)
        if not reasoning_match:
            return R_star
        reasoning_output = reasoning_match.group(1).strip() if reasoning_match else ""
        R_star_item = deduplicated_results + [reasoning_output] if reasoning_output else deduplicated_results
        R_star.extend(R_star_item)
        return R_star

        

def parse_args():
    parser = argparse.ArgumentParser(description="reasoning_output_induction")
    
    parser.add_argument('--model_path', type=str, default="xxx",
                       help='Path of the model (.pt file)')
    parser.add_argument('--dataset_path', type=str, default="xxx",
                       help='Path of the dataset')
    parser.add_argument('--domain', type=str, default="xxx",
                       choices=['gsm8k', 'mmlu', 'svamp', 'humaneval'],
                       help='Domain of the dataset')
    parser.add_argument('--llm_name', type=str, default='gpt-5',
                       help='Name of the LLM')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device (cuda:0, cpu)')
    parser.add_argument('--mode', type=str, default='FullConnected')
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                       help="['CodeWriting'] for humaneval, ['MathSolver'] for gsm8k and svamp, ['AnalyzeAgent'] for mmlu")
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                       help="[4] for humaneval, [4] for gsm8k and svamp, [5] for mmlu")
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                       help="FinalRefer for gsm8k, svamp and mmlu, FinalWriteCode for humaneval")
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--current_time', type=str, default=None,help="The current time")
    parser.add_argument('--model_name', type=str, default="GPT-5")
    
    return parser.parse_args()

def load_dataset(self, dataset_path: str, dataset_type: str = None) -> List[Dict[str, Any]]:

        if dataset_type is None:
            dataset_type = self.domain
        
        print(f"Loading {dataset_type} dataset: {dataset_path}")
        
        if dataset_type == "gsm8k":
            dataset = JSONLReader.parse_file(dataset_path)
            from datasets.gsm8k_dataset import gsm_data_process_adversial
            processed_data = gsm_data_process_adversial(dataset)
            return processed_data
        
        elif dataset_type == "svamp":
            dataset = JSONLReader.parse_file(dataset_path)
            from datasets.gsm8k_dataset import svamp_data_process_adversial
            processed_data = svamp_data_process_adversial(dataset)
            return processed_data
        
        elif dataset_type == "mmlu":
            from datasets.mmlu_dataset import MMLUDataset
            dataset_train = MMLUDataset('dev')
            dataset_val = MMLUDataset('val')
            dataset_val._total_df_adversial = dataset_val._total_df_adversial.head(100).reset_index(drop=True)
            return dataset_val._total_df_adversial
        
        elif dataset_type == "humaneval":
            dataset = JSONLReader.parse_file(dataset_path)
            return dataset

async def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_path, args.domain)
    induction = ReasoningOutputInduction(
        model_path=args.model_path,
        domain=args.domain,
        llm_name=args.llm_name,
        device=args.device,
        mode=args.mode,
        dataset=dataset,
        agent_names=args.agent_names,
        agent_nums=args.agent_nums,
        decision_method=args.decision_method,
        current_time=args.current_time,
        model_name=args.model_name,
        num_rounds=args.num_rounds,
    )



if __name__ == '__main__':
    asyncio.run(main())

