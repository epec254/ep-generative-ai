# Databricks notebook source
# MAGIC %run ./Gradio_App_Import

# COMMAND ----------

import gradio as gr
import uuid

with gr.Blocks() as demo:
    gr_state = gr.State(value={})
    with gr.Row():
        intro_md_str = """
        # Document QA
        ### Welcome to the Question-Answering Interface for the QA bot!
        """
        intro = gr.Markdown(intro_md_str)
    with gr.Row():
        input_box = gr.Textbox(
            label="Please provide your question",
            lines=3,
            value="How do you explode an array?",
        )
    with gr.Row():
        output = gr.Textbox(label="Answer:", lines=4)
    with gr.Row():
        # temperature_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.2)
        checkbox_use_mpt_llm = gr.Checkbox(label="Try Model Version B", value=False)
    with gr.Row():

        def ask_question(question, use_mpt_llm, state_history):
            my_uuid = str(uuid.uuid4())

            print(
                f"With use_mpt_llm {use_mpt_llm}. And generated UUID {my_uuid} Your question: {question}. state history {state_history}"
            )
            answer = "answer - call model api here"
            duration_ms = 3872
            num_tokens = 343

            state_history["uuid"] = my_uuid
            state_history["question"] = question
            state_history["answer"] = answer
            print(f"Received answer {answer}")

            display_str = answer
            return display_str

        submit_btn = gr.Button("Get Answer")
        submit_btn.click(
            fn=ask_question,
            inputs=[input_box, checkbox_use_mpt_llm, gr_state],
            outputs=output,
        )
    with gr.Row():

        def thumb_up(name, state_history):
            my_uuid = state_history["uuid"]
            question = state_history["question"]
            answer = state_history["answer"]

            print(f"Thumb up with uuid {my_uuid}, question {question}, answer {answer}")

            return "Thumbed up!", state_history

        def thumb_down(name, state_history):
            my_uuid = state_history["uuid"]
            question = state_history["question"]
            answer = state_history["answer"]

            print(
                f"Thumb down with uuid {my_uuid}, question {question}, answer {answer}"
            )

            return "Thumbed down!", state_history

        thumb_up_btn = gr.Button("👍")
        thumb_up_btn.click(fn=thumb_up, inputs=[input_box, gr_state], outputs=None)
        thumb_down_btn = gr.Button("👎")
        thumb_down_btn.click(fn=thumb_down, inputs=[input_box, gr_state], outputs=None)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load the application

# COMMAND ----------

# Accepts any `gradio.blocks.Blocks` and (optional) port (default 8800)
gradio_driver_proxy_app = GradioDriverProxyApp(demo)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Run it 
# MAGIC
# MAGIC This must be a seperate step to see the URL outputed above.

# COMMAND ----------

gradio_driver_proxy_app.run()
