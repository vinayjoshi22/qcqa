import os
import argparse
import numpy as np
import pandas as pd
from transformers.models.auto import AutoModelForCausalLM
from qcqa import Constants, QCQA

parser = argparse.ArgumentParser(description='Run QCQA on a model')
parser.add_argument('--model_name', type=str, default="facebook/opt-125m", help='Model name')
parser.add_argument('--num_heads', type=int, default=12, help='Number of heads')
parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
parser.add_argument('--num_groups', type=int, default=6, help='Number of groups')
parser.add_argument('--root_path', type=str, default="output", help='Output path')
parser.add_argument('--path_prefix', type=str, default="expt-1", help='Expt path')
parser.add_argument('--KV_parse_strings', type=str, default="layer.{}.attention.self.key.weight;layer.{}.attention.self.value.weight", help='Key/Value parse strings e.g., layer.{}.attention.self.key.weight;layer.{}.attention.self.value.weight')
parser.add_argument('--n_gen', type=int, default=10, help='Num of generations to run')
parser.add_argument('--pop_sz_algo1', type=int, default=100, help='Population size')
parser.add_argument('--crossover_prob_algo1', type=float, default=0.9, help='Crossover probability')
parser.add_argument('--mutation_prob_algo1', type=float, default=1/32, help='Mutation probability')
parser.add_argument('--pop_sz_algo2', type=int, default=100, help='Population size')
parser.add_argument('--crossover_prob_algo2', type=float, default=0.9, help='Crossover probability')
parser.add_argument('--mutation_prob_algo2', type=float, default=1/32, help='Mutation probability')
args = parser.parse_args()

num_heads = args.num_heads
num_layers = args.num_layers
num_groups = args.num_groups
model_name = args.model_name
kv_parse_strings = args.KV_parse_strings.split(";")

root_path = args.root_path
path_prefix = args.path_prefix
save_path = os.path.join(root_path, model_name, path_prefix)
os.makedirs(save_path, exist_ok=True)

grp_sz = int(num_heads//num_groups)
assert grp_sz*num_groups == num_heads, "Number of heads should be divisible by number of groups"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Loading checkpoint...")
model_checkpoint = model.state_dict()

KV_parse_strings = []
for i in range(num_layers):
        KV_parse_strings.append( [kv_parse_strings[0].format(i), kv_parse_strings[1].format(i)] )

assert len(KV_parse_strings) == num_layers, "KV parse strings not equal to 2*num_layers!"
assert all([((KV_parse_strings[i][0] in model_checkpoint.keys()) and (KV_parse_strings[i][1] in model_checkpoint.keys())) for i in range(num_layers)]), "Model checkpoint keys: {} \n do not contain the key/value weight keys: {}!".format(model_checkpoint.keys(), KV_parse_strings)

ga_config_algo1 = {"pop_sz": args.pop_sz_algo1, "crossover_prob": args.crossover_prob_algo1, "mutation_prob": args.mutation_prob_algo1}
ga_config_algo2 = {"pop_sz": args.pop_sz_algo1, "crossover_prob": args.crossover_prob_algo2, "mutation_prob": args.mutation_prob_algo2}
ga_config = {Constants.ALGO1: ga_config_algo1, Constants.ALGO2: ga_config_algo2}

print("Saving config...")
with open(os.path.join(save_path, "config.txt"), "w") as f:
    f.write("Model: {}\n".format(model_name))
    f.write("Number of heads: {}\n".format(num_heads))
    f.write("Number of layers: {}\n".format(num_layers))
    f.write("Number of groups: {}\n".format(num_groups))
    f.write("Root path: {}\n".format(root_path))
    f.write("Path prefix: {}\n".format(path_prefix))
    f.write("KV parse strings: {}\n".format(args.KV_parse_strings))
    f.write("Number of generations: {}\n".format(args.n_gen))
    f.write("Population size algo1: {}\n".format(args.pop_sz_algo1))
    f.write("Crossover probability algo1: {}\n".format(args.crossover_prob_algo1))
    f.write("Mutation probability algo1: {}\n".format(args.mutation_prob_algo1))
    f.write("Population size algo2: {}\n".format(args.pop_sz_algo2))
    f.write("Crossover probability algo2: {}\n".format(args.crossover_prob_algo2))
    f.write("Mutation probability algo2: {}\n".format(args.mutation_prob_algo2))
    
## Run algorithm 1
print("Running algorithm 1...")
qcqa = QCQA(num_heads, num_layers, num_groups, model_checkpoint, KV_parse_strings, root_path, ga_config)
results_algo1 = qcqa.run(algorithm=Constants.ALGO1, n_gen=args.n_gen, seed=1234)

## Save the results of algorithm 1
print("Saving results of algorithm 1...")
for iL, res in enumerate(results_algo1):
    arr = np.concatenate([res.X, res.F], axis=1)
    cols = ["L{}H{}".format(iL,j) for j in range(num_heads)] + ["KV Cache", "WSE"]
    df = pd.DataFrame(arr, columns=cols)
    df.to_csv(os.path.join(save_path, "algo1_result_layer_{}.csv".format(iL)), index=False)

## Run algorithm 2
print("Running algorithm 2...")
group_info = np.array([res.X[res.F[:,1].argmin(), :] for res in results_algo1])
qcqa.set_group_info(group_info=group_info)
results_algo2 = qcqa.run(algorithm=Constants.ALGO2, n_gen=args.n_gen, seed=1234)

# Save the results of algorithm 2
print("Saving results of algorithm 2...")
for res in results_algo2:
    arr = np.concatenate([res.X, res.F], axis=1)
    cols = ["L{}".format(i) for i in range(num_layers)] + ["KV Cache", "WSE"]
    df = pd.DataFrame(arr, columns=cols)
    df.to_csv(os.path.join(save_path, "algo2_result.csv"), index=False)

gqas_cache, gqas_wse = [], []
gqa_candidate = np.array([i//grp_sz for i in range(num_heads)]).reshape(1, num_heads)
for iL in range(num_layers):
    _gqa_cache = qcqa.fitness_kv_cache(gqa_candidate, iL) 
    _gqa_wse = qcqa.fitness_wse(gqa_candidate, iL)
    gqas_cache.append(_gqa_cache)
    gqas_wse.append(_gqa_wse)
gqas = np.concatenate([np.array(gqas_cache).T, np.array(gqas_wse).T], axis=0)

## Plot the results with respect to GQA for algorithm 1 from the paper
import matplotlib.pyplot as plt
for iL, res in enumerate(results_algo1):
    fig_path = os.path.join(save_path, "algo1_layer_{}".format(iL))
    plt.scatter(res.F[:, 0], res.F[:, 1], color='blue')
    plt.scatter(gqas[0, iL], gqas[1, iL], color='red')
    plt.xlabel("KV Cache")
    plt.ylabel("WSE")
    plt.title("KV Cache vs WSE")
    plt.savefig(fig_path)
    plt.close()

## Plot the results with respect to GQA for algorithm 2 from the paper
import matplotlib.pyplot as plt
for iL, res in enumerate(results_algo2):
    fig_path = os.path.join(save_path, "algo2_layer_{}".format(iL))
    plt.scatter(res.F[:, 0], res.F[:, 1], color='blue')
    plt.scatter(gqas[0].mean(axis=-1), gqas[1, iL].sum(axis=-1), color='red')
    plt.xlabel("KV Cache")
    plt.ylabel("WSE")
    plt.title("KV Cache vs WSE")
    plt.savefig(fig_path)
    plt.close()
