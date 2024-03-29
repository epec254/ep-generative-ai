# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks: MPT-7B-Instruct
# MAGIC
# MAGIC Notebook 2 of 2: First, start the driver proxy for the model you want to access.  Then use this this notebook to access the model hosted by a Driver Proxy.
# MAGIC
# MAGIC This notebook can be run on any cluster type, but tested on 13.0 MLR i3 node.

# COMMAND ----------

# MAGIC %pip install --upgrade langchain

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
import requests

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Copy / paste the code from the Driver Proxy notebook.

# COMMAND ----------

driver_proxy_api = 'https://NAME.cloud.databricks.com/driver-proxy-api/o/CLUSTER_ID/7777' # from the output from the langChain
cluster_id = 'xxx'
port = 7777

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Sample request via API call

# COMMAND ----------

ctx = get_context()

api_token = ctx.apiToken

def generate_params_to_json(generate_params):
  params = {}
  for key, value in generate_params:
    params[key] = value
  return params

# Supported parameters

# do_sample (bool, optional): Whether or not to use sampling. Defaults to True.
# max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 256.
# top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with
#     probabilities that add up to top_p or higher are kept for generation. Defaults to 1.0.
# top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
#     Defaults to 50.
# temperature (int, optional): Adjusts randomness of outputs, greater than 1 is random and 0.01 is deterministic. (minimum: 0.01; maximum: 5)
def generate_text(prompt, **generate_params):
  # json_request = generate_params_to_json(**generate_params)

  generate_params['prompt'] = prompt

  print(generate_params)

  resp = requests.post(driver_proxy_api, json=generate_params, headers={"Authorization": f"Bearer {api_token}"})
  if resp.ok:
      return resp.text
  raise

generate_text('hello, how are you', temperature=0.4)

# COMMAND ----------

from langchain.llms import Databricks

# Set extra parameters like `temperature` in `model_kwargs`.
llm = Databricks(cluster_id=cluster_id, cluster_driver_port=port, model_kwargs={"temperature":0.0000001})

llm("what is databricks?")
