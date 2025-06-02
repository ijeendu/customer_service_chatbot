# This is a repository for a chatbot that responds to customer service queries built using Ollama models for embedding and chat responses, Chromadb for the Vector database and Streamlit for the user interface.


# copy into your .env file:

```
LANGFUSE_SECRET_KEY="YOUR LANGFUSE SECRET KEY"
LANGFUSE_PUBLIC_KEY="YOUR LANGFUSE PUBLIC KEY"
LANGFUSE_HOST="https://us.cloud.langfuse.com"
```

To Use OLLama for depeval: https://medium.com/@jeffreyip54/you-can-now-use-ollama-for-llm-as-a-judge-76f06e3005c9


To get started, you’ll need to:

Install DeepEval (using pip)
Ollama for macOS, Linux, or Windows
Once you have everything setup, don’t forget to run your model:

ollama run deepseek-r1:1.5b
And set deepeval to use Ollama for evaluation in the CLI:

deepeval set-ollama deepseek-r1:1.5b
