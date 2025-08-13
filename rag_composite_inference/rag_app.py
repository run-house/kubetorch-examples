# # RAG App with Vector Embedding and LLM Generation
#
# Retrieval augmented generation (RAG) combines custom traditional information retrieval with large language models
# to enirch the response from the LLM. By embedding content from context-specific URLs into a vector database, the
# system can intelligently answer questions with relevant, sourced information.
#
# ## Overview
# In this example, we build a FastAPI application that extracts content from public URLs, embeds the text into a vector
# database, and uses those embeddings to generate accurate, context-aware answers to user questions.
#
# We use Kubetorch to deploy the underlying embedding and LLM components as standalone services that are easy to
# develop, debug, and scale independently. By decoupling accelerated compute tasks from the main application, the FastAPI
# app remains lightweight and easier to maintain, simply calling out to the remote services. The FastAPI app itself
# can also be deployed as a Kubetorch service within a Kubernetes environment.
#
# The application exposes two main endpoints:
# - Indexing (`/add`)
#      - Accept a list of URLs via POST endpoint
#      - Parse text from URLs using [LangChain](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)
#      - Extract and embed the content from the text using the Embedder service which uses [Sentence Transformers
#        (SBERT)](https://sbert.net/index.html)
#      - Store the embeddings in a vector database ([LanceDB](https://lancedb.com/))
#
# - Retrieval and Generation (`/generate`)
#      - Accept a question via GET endpoint
#      - Create a vector embedding from the question using the Embedder service and retrieve related documents from the vector database
#      - Construct an LLM prompt from the documents and the original question
#      - Generate a response using the LLM (Llama 3) service
#      - Output the response with source URLs and question input
#
# ![Graphic displaying the steps of indexing data and the retrieval and generation process](https://runhouse-tutorials.s3.amazonaws.com/indexing-retrieval-generation.png)
#
# Note: Some of the steps in this example could also be accomplished with platforms like OpenAI and
# tools such as LangChain, but we break out the components explicitly to fully illustrate each step and make the
# example easily adaptible to other use cases. Swap out components as you see fit!
#
# ## Deploying the Vector Embedding and LLM Services
from typing import Dict, List, Union

import kubetorch as kt
#
# ### Create Vector Embedding Service
# We construct a standard Embedder class, and use kubetorch decorators to convert it into a deployable kubetorch service.
# In the `@kt.compute` decorator, we specify the number of GPUs to use, as well as the base image of the service. We use
# the PyTorch docker image and additionally install the `sentence_transformers` package on top of it.
#

embedder_img = kt.Image(image_id="nvcr.io/nvidia/pytorch:25.05-py3").pip_install(
    ["sentence_transformers"]
)


@kt.compute(gpus="1", image=embedder_img)
@kt.autoscale(initial_replicas=1, min_replicas=0, max_replicas=4, concurrency=1000)
class Embedder:
    def __init__(self, model_name_or_path="BAAI/bge-large-en-v1.5", **model_kwargs):
        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            model_name_or_path=model_name_or_path, device="cuda", **model_kwargs
        )
        self.model = torch.compile(model)

    def embed(self, text: Union[List, str], **embed_kwargs):
        if not isinstance(text, list):
            text = [text]
        return self.model.encode(text, **embed_kwargs).tolist()


# ### Define the LLM (LLama) Service
# We similarly define a `LlamaModel` class that uses kubetorch decorators to convert it into a deployable service.
# The `LlamaModel` class is detailed in the [LLM Inference Example](https://www.run.house/examples/vllm-llama3-inference), so
# here we simply reimport the `LlamaModel` class from that example. For more details on the `LlamaModel` implementation,
# see that example.

from vllm_inference.llama import LlamaModel


# ### Deploy the Services with Kubetorch
# Now that we have defined the `Embedder` and `LlamaModel` classes with the `@kt.compute` decorator, we can deploy them
# with kubetorch by using the cli command `kt deploy rag_composite_inference/rag_app.py`. This will deploy the classes
# as remote services on Kubernetes.
#
# ```bash
# kt deploy rag_app.py
# ```
#
# Once deployed, you can access the services by directly importing the class in your program, and use any methods of the
# class.
#
# ```python
# from rag_app import Embedder, LlamaModel
#
# res = Embedder.embed("This is a test sentence.", normalize_embeddings=True)
# print(res)
# ```
#
# Alternatively, to debug locally, you can also deploy the services in Python by running `<class_name>.deploy()` for each
# class.
#
# ```python
# if __name__ == "__main__":
#     Embedder.deploy()
#     LlamaModel.deploy()
#
#     res = Embedder.embed("This is a test sentence.", normalize_embeddings=True)
#     print(res)
# ```
#
# ## Defining the FastAPI App
# Now that we have the `Embedder` and `LlamaModel` services up and running, we define the FastAPI app, which calls to
# these services inside it's add document and generate response endpoints. The app uses a LanceDB vector database
# to store the embeddings.
#
# ### RAG App Setup
# First, we'll import the necessary packages to define and run the FastAPI app.

import uvicorn
from fastapi import Body, FastAPI, HTTPException

app = FastAPI()

# We also import the the LanceDB packages necessary for the vector database, and set up the LanceDB client and table.
# We wrap the LanceDB imports in a try block, as they are not required locally if we are deploying the app as a
# Kubetorch service.

try:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector

    class Item(LanceModel):
        url: str
        page_content: str
        vector: Vector(1024)

except ImportError:
    pass

lance_client = None
lance_db = None


def get_lance_db():
    """Get or create the LanceDB client and table"""
    global lance_client, lance_db

    if lance_client is None:
        lance_client = lancedb.connect("/tmp/db")
        lance_db = lance_client.create_table(
            "rag-table", schema=Item.to_arrow_schema(), exist_ok=True
        )

    return lance_db


# We also define the prompt template for the LLM generation phase of the RAG app.
PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer: """

# To leverage the documents retrieved from the previous step, we'll format a prompt that provides text from
# related documents as "context" for the LLM. This allows a general purpose LLM (like Llama) to provide
# more specific responses to a particular question.
async def format_prompt(text: str, docs: List[Dict]) -> str:
    """Retrieve documents from vector DB related to input text"""
    context = "\n".join([doc["page_content"] for doc in docs])
    prompt = PROMPT_TEMPLATE.format(question=text, context=context)
    return prompt


# ### Defining the Kubetorch App
# Here we define a Kubetorch app using `kt.app()`, which will deploy the FastAPI app as a remote Kubernetes
# service named `ragapp` if the file is run with the `kt run` CLI command. The app specifies the compute and
# requirements necessary to run the server, as well as which port to run it on. If you want to run the app
# locally, simply run the file as you would normally, which will skip the deployment step.

lance_fastapi_image = kt.Image().pip_install(
    [
        "lancedb>=0.3.0",
        "langchain_community",
        "langchain_text_splitters",
        "fastapi[standard]",
        "bs4",
    ]
)
kt_app = kt.app(
    name="ragapp",
    cpus="0.01",
    port=8000,
    health_check="/health",
    image=lance_fastapi_image,
)

# ### Defining the Endpoints
#
# Add an endpoint to check on our app health. This is a minimal function intended to only
# show if the application is up and running or down.
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# #### Vector Embedding POST Endpoint
# This method uses the embedder service to create database entries with the source, content, and vector embeddings
# for chunks of text from a provided list of URLs.
@app.post("/add")
async def add_document(paths: List[str] = Body([]), kwargs: Dict = Body({})):
    """Generate embeddings for the URL and write to DB."""
    try:
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Load and split documents
        docs = WebBaseLoader(web_paths=paths).load()
        split_docs = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=50
        ).split_documents(docs)
        splits_as_str = [doc.page_content for doc in split_docs]

        # Generate embeddings using the remote Embedder service
        embeddings = Embedder.embed(
            splits_as_str, normalize_embeddings=True, stream_logs=False, **kwargs
        )

        # Create items for LanceDB
        items = [
            Item(
                url=doc.metadata["source"],
                page_content=doc.page_content,
                vector=embeddings[index],
            )
            for index, doc in enumerate(split_docs)
        ]

        # Add to LanceDB
        db = get_lance_db()
        db.add(items)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed URLs: {str(e)}")


# #### Generation GET endpoint
# This endpoint runs inference on our LLM to generate a response to a question. The results are enhanced by first
# retrieving related documents from the Vector DB, which contains source URLs fed into the POST endpoint. Content from
# the fetched documents is then formatted into the text prompt sent to our self-hosted LLM. We use the generic prompt
# template defined in the setup section above.
@app.get("/generate")
async def generate_response(text: str, limit: int = 4):
    """Generate a response to a question using an LLM with context from our database"""
    if not text:
        return {"error": "Question text is missing"}

    try:
        # Encode the input text into a vector using the remote Embedder service
        vector = Embedder.embed(
            text=[text], normalize_embeddings=True, stream_logs=False
        )[0]

        # Search LanceDB for nearest neighbors to the vector embed
        db = get_lance_db()
        documents = db.search(vector).limit(limit).to_list()

        # List of sources from retrieved documents
        sources = set([doc["url"] for doc in documents])

        # Create a prompt using the documents and search text
        prompt = await format_prompt(text, documents)

        # Send prompt with optional sampling parameters for vLLM
        # More info: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
        response = LlamaModel.generate(
            queries=prompt,
            temperature=0.8,
            top_p=0.95,
            max_tokens=100,
            stream_logs=False,
        )

        return {"question": text, "response": response, "sources": sources}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate response: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ## Running the App
# ### Locally
# You can use the following command to run the app locally from your terminal:
#
# ```bash
# fastapi run rag_composite_inference/rag_app.py
# ```
# This creates a local FastAPI server that you can use to test the app, and iterate on the app or services.
# You can interact with the app locally by making requests to the endpoints using curl or Postman, as you normally
# would with a local FastAPI app.
#
# ### Remotely
# After making any necessary changes to the app and ensuring that it is working as intended, you can deploy the app as
# a Kubetorch service by running:
#
# ```bash
# kt run fastapi run rag_composite_inference/rag_app.py
# ```
#
# To run queries to the remote app locally, create a port forward to the remote service using the `kt port-forward` command:
#
# ```bash
# kt port-forward ragapp <local_port>
# ```
#
# Then run queries to the remote app as you normally would locally with curl, just adding the additonal `http/` prefix
# to the endpoint to signal that you are making a request to the remote app server, rather than Kubetorch's server:
#
# ```bash
# curl --header "Content-Type: application/json" \
#   --request POST \
#   --data '{"paths":["https://www.nps.gov/yell/planyourvisit/safety.htm", "https://en.wikipedia.org/wiki/Adventures_of_the_Gummi_Bears"]}' \
#   http://localhost:32300/http/add
# ```
#
# ```text
# http://localhost:32300/http/generate?text=Does%20yellowstone%20have%20gummi%20bears%3F
# ```
#
