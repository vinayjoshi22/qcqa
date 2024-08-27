import os

num_groups = [2,4,6]
KV_parse_strings = "model.decoder.layers.{}.self_attn.k_proj.weight;model.decoder.layers.{}.self_attn.v_proj.weight"
model = "facebook/opt-125m"
for num_group in num_groups:
    cmd = "python expt.py --model {} --num_groups {} --path_prefix num_groups_{} --KV_parse_strings {} --num_layers 12 --num_heads 12 ".format(model, num_group, num_group, KV_parse_strings)
    print("Running command: {}".format(cmd))
    ret = os.system(cmd)
    status = "Success" if ret == 0 else "Failed"
    print("Status: {}".format(status))
