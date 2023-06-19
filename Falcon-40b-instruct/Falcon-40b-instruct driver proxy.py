# Databricks notebook source
# MAGIC %md
# MAGIC g5.12xlarge: 4 A10 GPUs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------

# MAGIC %pip install --upgrade torch==2.0.* einops

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# cache mpt-7b on DBFS to avoid re-download on restart of cluster

%env TRANSFORMERS_CACHE=/dbfs/ep/cache
%env HF_HOME=/dbfs/ep/cache
%env HF_HUB_DISABLE_SYMLINKS_WARNING=TRUE
%env HF_DATASETS_CACHE=/dbfs/ep/cache

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    revision="1e7fdcc9f45d13704f3826e99937917e007cd975" # cached as of 6/16
)

display('model loaded')

# COMMAND ----------

def falcon40_instruct_generate(prompt, **generate_params):
  if 'max_new_tokens' not in generate_params:
    generate_params['max_new_tokens'] = 256

  if 'temperature' not in generate_params:
    generate_params['temperature'] = 1.0

  if 'top_k' not in generate_params:
    generate_params['top_k'] = 50
  if 'eos_token_id' not in generate_params:
    generate_params['eos_token_id'] = tokenizer.eos_token_id
    generate_params['pad_token_id'] = tokenizer.eos_token_id
  
  if 'do_sample' not in generate_params:
    generate_params['do_sample'] = True
  
  generate_params['use_cache'] = True

  if 'num_return_sequences' not in generate_params:
    generate_params['num_return_sequences'] = 1

  sequences = pipeline(
    prompt,
    **generate_params,
  )

  return sequences[0]['generated_text']

# COMMAND ----------

print(falcon40_instruct_generate("What is Databricks?"))

# COMMAND ----------

print(falcon40_instruct_generate("What is Databricks?", max_new_tokens=1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve with Flask

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("falcon-40b-instruct")

@app.route('/', methods=['POST'])
def serve_falcon_40b_instruct():
  resp = falcon40_instruct_generate(**request.json)
  return jsonify(resp)

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7777"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}"

print(f"""
driver_proxy_api = '{driver_proxy_api}'
cluster_id = '{ctx.clusterId}'
port = {port}
""")

# COMMAND ----------

app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
