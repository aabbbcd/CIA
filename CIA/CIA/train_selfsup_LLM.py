import argparse
import os
from trainer_LLM import SelfSupervisedTrainer
from data_processor import GraphDataProcessor
import json
import logging
import datetime
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
	p = argparse.ArgumentParser()

	p.add_argument('--data_path', type=str, default="",help="Path of grount-truth graph execution results")
	p.add_argument('--outputs_path', type=str, default="",help="Path of reasoning output induction results")
	p.add_argument('--domain', type=str, default="",choices=['gsm8k', 'mmlu', 'svamp', 'humaneval'],help="Domain of the dataset")
	p.add_argument('--device', type=str, default="cuda:0")
	p.add_argument('--text_encoder', type=str, default='/data/llm/all-MiniLM-L6-v2')
	p.add_argument('--hidden_dim', type=int, default=256)
	p.add_argument('--dropout', type=float, default=0.1)
	p.add_argument('--tokenizer_max_length', type=int, default=128)
	p.add_argument('--lr', type=float, default=0.0002)
	p.add_argument('--weight_decay', type=float, default=1e-4)
	p.add_argument('--edge_threshold', type=float, default=0.5)
	p.add_argument('--epochs', type=int, default=50)
	p.add_argument('--encoder', type=str, default='sentence_transformers')
	now_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
	p.add_argument('--log_path', type=str, default=f"train_{now_str}.log")
	return p.parse_args()

def main():
	args = parse_args()
	logger = logging.getLogger("selfsup")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	fh = logging.FileHandler(args.log_path, mode='a')
	fh.setLevel(logging.INFO)
	fh.setFormatter(fmt)
	sh = logging.StreamHandler()
	sh.setLevel(logging.INFO)
	sh.setFormatter(fmt)
	logger.addHandler(fh)
	logger.addHandler(sh)

	processor = GraphDataProcessor()
	graphs = processor.process_all_graphs(args.data_path)

	auc=0
	f1=0
	acc=0
	counter=0
	max_auc=0

	logger.info(json.dumps(vars(args), ensure_ascii=False))
	
	for idx, g in enumerate(graphs):
		with open(f"{args.outputs_path}/R_stars_{idx}.json", "r") as f:
			R_stars = json.load(f)
		trainer = SelfSupervisedTrainer(
		data_path=args.data_path, 
		outputs=R_stars,
		encoder=args.encoder,
		text_encoder_name=args.text_encoder,
		device=args.device,
		hidden_dim=args.hidden_dim,
		dropout=args.dropout,
		tokenizer_max_length=args.tokenizer_max_length,
		lr=args.lr,
		weight_decay=args.weight_decay,
		edge_threshold=args.edge_threshold,
		total_steps=args.epochs,
		max_agents=g["num_nodes"],
	)
		aucs=[]
		for epoch in range(1, args.epochs + 1):
			print(counter)
			eval_log = trainer.train_epoch(g)
			if eval_log['auc'] > max_auc:
				max_acc = eval_log['accuracy']
				max_f1=eval_log['f1']
				max_auc = eval_log['auc']
			aucs.append(eval_log['auc'])
		logger.info(f"auc={aucs}")
		auc += max_auc
		acc += max_acc
		f1 += max_f1
		counter += 1
logger.info(f'auc={auc/counter} f1={f1/counter} acc={acc/counter}')


if __name__ == '__main__':
	main()
