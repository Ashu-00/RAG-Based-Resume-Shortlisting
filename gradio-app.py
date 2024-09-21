import gradio as gr
import requests

# Define the API endpoint
API_URL = "http://localhost:8000/query"


def query_api(text, topk):
    # Payload for the API call
    
    payload = {
        "text": text,
        "top_k": int(topk),
    }

    # Make a POST request to the FastAPI backend
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract the answer and sources from the response
        answer = data["answer"]
        sources = "\n".join(data["sources"])

        return answer, sources

    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"


# Create a Gradio interface
iface = gr.Interface(
    fn=query_api,  # The function that processes the input
    inputs=[
        gr.Textbox(label="Describe the role"),
        gr.Number(label="Top K results", value=5, precision=0),
    ],
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Textbox(label="Document Sources"),
    ],
    title="RAG Pipeline Query",
    description="Enter the type of role and description.",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
