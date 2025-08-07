import gradio as gr
from dotenv import load_dotenv
from handle_file_ingestion import (
    file_upload_handler,
    get_uploaded_files,
)
from handle_url_ingestion import (
    url_upload_handler,
    get_uploaded_urls,
)
from manage_data import delete_document, delete_url, clear_all_data
from handle_chat import handle_chat

load_dotenv()


# --- Tab 1: Upload Documents UI ---
def upload_ui():
    with gr.Row():
        file_input = gr.File(label="Upload File", file_types=[".pdf", ".md", ".txt"])
    output = gr.Textbox(label="Status")

    upload_button = gr.Button("Ingest File")
    upload_button.click(fn=file_upload_handler, inputs=file_input, outputs=output)

    return file_input, upload_button, output


# --- Tab 2: Upload URLs UI ---
def url_upload_ui():
    with gr.Row():
        url_input = gr.Textbox(label="Enter URL", placeholder="https://example.com")
    output = gr.Textbox(label="Status")

    upload_button = gr.Button("Ingest URL")
    upload_button.click(fn=url_upload_handler, inputs=url_input, outputs=output)

    return url_input, upload_button, output


# --- Tab 3: Document Management UI ---
def manage_ui():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Files")
            file_dropdown = gr.Dropdown(choices=[], label="Uploaded Files")
            delete_file_btn = gr.Button("Delete Selected File")
            refresh_files_btn = gr.Button("Refresh File List")
            file_delete_output = gr.Textbox(
                label="File Delete Status", lines=2, visible=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### URLs")
            url_dropdown = gr.Dropdown(choices=[], label="Ingested URLs")
            delete_url_btn = gr.Button("Delete Selected URL")
            refresh_urls_btn = gr.Button("Refresh URL List")
            url_delete_output = gr.Textbox(
                label="URL Delete Status", lines=2, visible=False
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Nuclear Option - Clear All Data")
            gr.Markdown(
                "⚠️ **Warning:** This will delete ALL uploaded files, ingested URLs, and clear the entire vector database. This action cannot be undone."
            )
            clear_all_btn = gr.Button("Clear All Data", variant="stop")

    def update_file_dropdown_choices():
        return gr.update(choices=[f[0] for f in get_uploaded_files()])

    def update_url_dropdown_choices():
        return gr.update(choices=get_uploaded_urls())

    def show_file_status(result):
        return gr.update(value=result, visible=True)

    def show_url_status(result):
        return gr.update(value=result, visible=True)

    delete_file_btn.click(
        fn=delete_document, inputs=[file_dropdown], outputs=[file_delete_output]
    )
    delete_file_btn.click(
        fn=show_file_status, inputs=[file_delete_output], outputs=[file_delete_output]
    )
    delete_file_btn.click(
        fn=update_file_dropdown_choices, inputs=[], outputs=[file_dropdown]
    )

    delete_url_btn.click(
        fn=delete_url, inputs=[url_dropdown], outputs=[url_delete_output]
    )
    delete_url_btn.click(
        fn=show_url_status, inputs=[url_delete_output], outputs=[url_delete_output]
    )
    delete_url_btn.click(
        fn=update_url_dropdown_choices, inputs=[], outputs=[url_dropdown]
    )

    clear_all_btn.click(fn=clear_all_data, inputs=[], outputs=[])
    clear_all_btn.click(
        fn=update_file_dropdown_choices, inputs=[], outputs=[file_dropdown]
    )
    clear_all_btn.click(
        fn=update_url_dropdown_choices, inputs=[], outputs=[url_dropdown]
    )

    refresh_files_btn.click(
        fn=update_file_dropdown_choices, inputs=[], outputs=[file_dropdown]
    )
    refresh_urls_btn.click(
        fn=update_url_dropdown_choices, inputs=[], outputs=[url_dropdown]
    )

    return (
        file_dropdown,
        delete_file_btn,
        file_delete_output,
        update_file_dropdown_choices,
        url_dropdown,
        delete_url_btn,
        refresh_files_btn,
        refresh_urls_btn,
        update_url_dropdown_choices,
    )


# --- Tab 4: Ask Questions UI---
def chat_ui():
    with gr.Row():
        gr.Markdown(
            """
        ## 🤖 Ask Questions About Your Documents
        
        Upload documents or ingest URLs in the other tabs, then come back here to ask questions about them!
        
        The AI will search through your uploaded documents and provide answers based on the content.
        """
        )

    # Create the chat interface
    chat_interface = gr.ChatInterface(
        fn=handle_chat,
        title="Q&A Chat",
        description="Ask questions about your uploaded documents and URLs",
        examples=[
            "What documents have been uploaded?",
            "Summarize the main points from the documents",
            "What are the key concepts discussed?",
            "Can you explain the technical details?",
        ],
    )

    return chat_interface


# --- Main Interface ---
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Upload Document"):
            upload_ui()
        with gr.TabItem("Ingest from URL"):
            url_upload_ui()
        with gr.TabItem("Manage Data"):
            (
                file_dropdown,
                delete_file_btn,
                delete_output,
                update_file_choices_fn,
                url_dropdown,
                delete_url_btn,
                refresh_files_btn,
                refresh_urls_btn,
                update_url_choices_fn,
            ) = manage_ui()
        with gr.TabItem("Ask Questions"):
            chat_ui()
    # Update dropdowns when app loads
    demo.load(fn=update_file_choices_fn, inputs=[], outputs=[file_dropdown])
    demo.load(fn=update_url_choices_fn, inputs=[], outputs=[url_dropdown])

demo.launch()

# Add a callback to run cleanup when the app shuts down
demo.close(fn=clear_all_data)
