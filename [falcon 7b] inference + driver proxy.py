# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Falcon-7B-Instruct
# MAGIC
# MAGIC This notebook enables you to run Falcon-7B-Instruct on a Databricks cluster and expose the model to LangChain or API via [driver proxy](https://python.langchain.com/en/latest/modules/models/llms/integrations/databricks.html#wrapping-a-cluster-driver-proxy-app).
# MAGIC
# MAGIC ## Instance type required
# MAGIC *Tested on* g5.4xlarge: 1 A10 GPUs
# MAGIC
# MAGIC Requires MLR 13.1+ and single node A10G GPU instance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------

# MAGIC %pip install --upgrade torch==2.0.* einops

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# cache on DBFS to avoid re-download on restart of cluster

%env TRANSFORMERS_CACHE=/dbfs/ep/cache
%env HF_HOME=/dbfs/ep/cache
%env HF_HUB_DISABLE_SYMLINKS_WARNING=TRUE
%env HF_DATASETS_CACHE=/dbfs/ep/cache

# COMMAND ----------

import transformers
import mlflow
import torch
from flask import Flask, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import snapshot_download

dbutils.fs.mkdirs(os.environ['HF_HOME'])
dbutils.fs.mkdirs(os.environ['TRANSFORMERS_CACHE'])
dbutils.fs.mkdirs(os.environ['HF_DATASETS_CACHE'])

# COMMAND ----------

torch.cuda.empty_cache()

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    revision="9f16e66a0235c4ba24e321e3be86dd347a7911a0" # cached as of 6/16
)

print(tokenizer.eos_token_id)

display('model loaded')

# COMMAND ----------

def falcon7_instruct_generate(prompt, **generate_params):
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

print(falcon7_instruct_generate("What is Databricks?"))

# COMMAND ----------

print(falcon7_instruct_generate("What is Databricks?", max_new_tokens=1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve with Flask

# COMMAND ----------

from flask import Flask, jsonify, request

app = Flask("falcon-40b-instruct")

@app.route('/', methods=['POST'])
def serve_falcon_7b_instruct():
  resp = falcon7_instruct_generate(**request.json)
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

# COMMAND ----------


