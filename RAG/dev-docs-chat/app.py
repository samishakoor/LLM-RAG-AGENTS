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
def file_upload_ui():
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                """
            ## 📁 File Upload Instructions
            
            **How to upload documents:**
            
            1. **Select File**: Choose a PDF, TXT, or Markdown file from your computer
            2. **Click Upload**: The file will be processed and embedded into the knowledge base
            3. **Wait for Processing**: The system will extract text, chunk it, and create embeddings
            4. **Check Status**: Look for the green ✅ success message when complete
            
            **Supported Formats:**
            - 📄 PDF files (.pdf)
            - 📝 Text files (.txt)
            - 📋 Markdown files (.md)
            - ❌ No duplicate file upload allowed
            
            **What happens next:**
            - Your document will be searchable in the "Ask Questions" tab
            - You can manage uploaded files in the "Manage Data" tab
            - Multiple files can be uploaded to build a comprehensive knowledge base
            """
            )

        with gr.Column(scale=7):
            file_input = gr.File(
                label="Upload File", file_types=[".pdf", ".md", ".txt"]
            )
            output = gr.Textbox(label="Status")

            upload_button = gr.Button("📤 Ingest File", variant="primary")
            upload_button.click(
                fn=file_upload_handler, inputs=file_input, outputs=output
            )

    return file_input, upload_button, output


# --- Tab 2: Upload URLs UI ---
def url_upload_ui():
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                """
            ## 🌐 URL Ingestion Instructions
            
            **How to ingest web content:**
            
            1. **Enter URL**: Paste a valid HTTPS URL (e.g., https://example.com)
            2. **Click Ingest**: The webpage will be scraped and processed
            3. **Wait for Processing**: Content will be extracted, chunked, and embedded
            4. **Check Status**: Look for the green ✅ success message when complete
            
            **URL Requirements:**
            - ✅ Must start with `https://`
            - ✅ Must be a valid, accessible webpage
            - ✅ Should contain readable text content
            - ❌ No duplicate URLs allowed
            
            **What happens next:**
            - Web content becomes searchable in the "Ask Questions" tab
            - You can manage ingested URLs in the "Manage Data" tab
            - Multiple URLs can be ingested to build a comprehensive knowledge base
            
            **Tips:**
            - Use documentation sites, articles, or blog posts for best results
            - Avoid social media or dynamic JavaScript-heavy sites
            - Large pages may take longer to process
            """
            )

        with gr.Column(scale=7):
            url_input = gr.Textbox(label="Enter URL", placeholder="https://example.com")
            output = gr.Textbox(label="Status")

            upload_button = gr.Button("🌐 Ingest URL", variant="primary")
            upload_button.click(fn=url_upload_handler, inputs=url_input, outputs=output)

    return url_input, upload_button, output


# --- Tab 3: Document Management UI ---
def manage_data_ui():
    with gr.Tab("Manage Data"):
        gr.Markdown("# 🗂️ Data Management")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Uploaded Files")
                file_dropdown = gr.Dropdown(
                    label="Select File to Delete",
                    choices=get_uploaded_files(),
                    interactive=True,
                )
                delete_file_btn = gr.Button("🗑️ Delete Selected File", variant="stop")
                refresh_files_btn = gr.Button("🔄 Refresh File List")
                file_delete_output = gr.Textbox(
                    label="File Delete Result", visible=False
                )

                def delete_selected_file(filename):
                    if filename:
                        result = delete_document(filename)
                        # Refresh the dropdown
                        new_choices = get_uploaded_files()
                        return gr.update(value=result, visible=True), gr.update(
                            choices=new_choices, value=None
                        )
                    return (
                        gr.update(value="No file selected", visible=True),
                        gr.update(),
                    )

                delete_file_btn.click(
                    delete_selected_file,
                    inputs=file_dropdown,
                    outputs=[file_delete_output, file_dropdown],
                )

                refresh_files_btn.click(
                    lambda: gr.update(choices=get_uploaded_files()),
                    outputs=file_dropdown,
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🌐 Ingested URLs")
                url_dropdown = gr.Dropdown(
                    label="Select URL to Delete",
                    choices=get_uploaded_urls(),
                    interactive=True,
                )
                delete_url_btn = gr.Button("🗑️ Delete Selected URL", variant="stop")
                refresh_urls_btn = gr.Button("🔄 Refresh URL List")
                url_delete_output = gr.Textbox(label="URL Delete Result", visible=False)

                def delete_selected_url(url):
                    if url:
                        result = delete_url(url)
                        # Refresh the dropdown
                        new_choices = get_uploaded_urls()
                        return gr.update(value=result, visible=True), gr.update(
                            choices=new_choices, value=None
                        )
                    return gr.update(value="No URL selected", visible=True), gr.update()

                delete_url_btn.click(
                    delete_selected_url,
                    inputs=url_dropdown,
                    outputs=[url_delete_output, url_dropdown],
                )

                refresh_urls_btn.click(
                    lambda: gr.update(choices=get_uploaded_urls()), outputs=url_dropdown
                )

        gr.Markdown("---")
        gr.Markdown("### ⚠️ Nuclear Option - Clear All Data")
        gr.Markdown(
            "**Warning**: This will delete ALL uploaded files, ingested URLs, and clear the entire vector database. This action cannot be undone."
        )

        with gr.Row():
            with gr.Column():
                clear_all_btn = gr.Button(
                    "💥 Clear All Data", variant="stop", size="lg"
                )
                clear_output = gr.Textbox(label="Clear All Result", visible=False)

        def clear_all_data_wrapper():
            try:
                clear_all_data()
                return gr.update(
                    value="✅ All data cleared successfully!", visible=True
                )
            except Exception as e:
                return gr.update(
                    value=f"❌ Error clearing data: {str(e)}", visible=True
                )

        clear_all_btn.click(clear_all_data_wrapper, outputs=clear_output)

        # Load initial data
        demo.load(
            fn=lambda: gr.update(choices=get_uploaded_files()), outputs=file_dropdown
        )
        demo.load(
            fn=lambda: gr.update(choices=get_uploaded_urls()), outputs=url_dropdown
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
        type="messages",
        title="💬 Q&A Chat",
        description="Ask questions about your uploaded documents and URLs",
        examples=[
            "Help me understand this topic like I'm new to it",
            "Summarize the main points from the documents",
            "What are the key concepts discussed?",
            "Can you explain the technical details?",
        ],
    )
    return chat_interface


# --- Main Interface ---
with gr.Blocks(title="Developer Docs Assistant") as demo:
    gr.Markdown(
        """
    # 📘 **Developer Docs Assistant**
    """
    )
    with gr.Tabs():
        with gr.TabItem("Upload Document"):
            file_upload_ui()
        with gr.TabItem("Ingest from URL"):
            url_upload_ui()
        with gr.TabItem("Manage Data"):
            manage_data_ui()
        with gr.TabItem("Ask Questions"):
            chat_ui()

demo.launch()

# Add a callback to run cleanup when the app shuts down
demo.close(fn=clear_all_data)
