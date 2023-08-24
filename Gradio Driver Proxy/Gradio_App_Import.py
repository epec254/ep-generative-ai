# Databricks notebook source
# MAGIC %pip install --upgrade --quiet databricks-cli==0.17.7
# MAGIC %pip install --quiet gradio fastapi uvicorn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
from dataclasses import dataclass
import gradio

import uvicorn
from fastapi import FastAPI

import nest_asyncio

@dataclass
class ProxySettings:
    proxy_url: str
    port: str
    url_base_path: str


class GradioDriverProxyApp:
    def __init__(self, gradio_blocks: gradio.blocks.Blocks, port: int = 8800):
        # self._app = data_app
        self._port = port
        import IPython

        self._dbutils = IPython.get_ipython().user_ns["dbutils"]
        self._display_html = IPython.get_ipython().user_ns["displayHTML"]
        self._context = json.loads(
            self._dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .toJson()
        )
        # need to do this after the context is set
        self._cloud = self.get_cloud()
        # create proxy settings after determining the cloud
        self._ps = self.get_proxy_settings()
        self._fastapi_app = self._make_fastapi_app(
            root_path=self._ps.url_base_path.rstrip("/")
        )
        self._streamlit_script = None
        # after everything is set print out the url

        self._gradio_blocks = gradio_blocks

        self.mount_gradio_app()

        # self.display_url(self.get_gradio_ahref_link())

    def _make_fastapi_app(self, root_path) -> FastAPI:
        fast_api_app = FastAPI(root_path=root_path)

        @fast_api_app.get("/")
        def read_main():
            return {
                "routes": [
                    {"method": "GET", "path": "/", "summary": "Landing"},
                    {"method": "GET", "path": "/status", "summary": "App status"},
                    {
                        "method": "GET",
                        "path": "/dash",
                        "summary": "Sub-mounted Dash application",
                    },
                ]
            }

        @fast_api_app.get("/status")
        def get_status():
            return {"status": "ok"}

        return fast_api_app

    def get_proxy_settings(self) -> ProxySettings:
        if self._cloud.lower() not in ["aws", "azure"]:
            raise Exception("only supported in aws or azure")

        org_id = self._context["tags"]["orgId"]
        org_shard = ""
        # org_shard doesnt need a suffix of "." for dnsname its handled in building the url
        if self._cloud.lower() == "azure":
            org_shard_id = int(org_id) % 20
            org_shard = f".{org_shard_id}"
        cluster_id = self._context["tags"]["clusterId"]
        url_base_path = f"/driver-proxy/o/{org_id}/{cluster_id}/{self._port}/"

        from dbruntime.databricks_repl_context import get_context

        host_name = get_context().browserHostName
        proxy_url = (
            f"https://{host_name}/driver-proxy/o/{org_id}/{cluster_id}/{self._port}/"
        )

        return ProxySettings(
            proxy_url=proxy_url, port=self._port, url_base_path=url_base_path
        )

    @property
    def app_url_base_path(self):
        return self._ps.url_base_path

    def mount_gradio_app(self, gradio_app=None):
        import gradio as gr

        if gradio_app is not None:
            self._gradio_blocks = gradio_app
        gr.mount_gradio_app(self._fastapi_app, self._gradio_blocks, f"/gradio")
        # self._fastapi_app.mount("/gradio", gradio_app)
        self.display_url(self.get_gradio_ahref_link())

    def get_cloud(self):
        if self._context["extraContext"]["api_url"].endswith("azuredatabricks.net"):
            return "azure"
        return "aws"

    def get_gradio_ahref_link(self):
        # must end with a "/" for it to not redirect
        return f'<a href="{self._ps.proxy_url}gradio/">Click to go to Gradio App @ {self._ps.proxy_url}gradio/</a>'

    def get_raw_gradio_url(self):
        # must end with a "/" for it to not redirect
        return f"{self._ps.proxy_url}gradio/"

    def display_url(self, url):
        self._display_html(url)

    def run(self):
        nest_asyncio.apply()
        uvicorn.run(self._fastapi_app, host="0.0.0.0", port=self._port)

    def run_gradio_app(self, gradio_app):
        self.mount_gradio_app(gradio_app)
        self.run()
