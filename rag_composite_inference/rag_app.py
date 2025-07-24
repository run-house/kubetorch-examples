# ## RAG App with Vector Embedding and LLM Generation
#
# This example defines a retrieval augmented generation (RAG) app that references text from websites
# to enrich the response from an LLM. Depending on the URLs you use to populate the vector database,
# you'll be able to answer questions more intelligently with relevant context.
#
# ### Example Overview
# Deploy a FastAPI app that is able to create and store embeddings from text on public website URLs,
# and generate answers to questions using related context from stored websites and an open source LLM.
#
# #### What does Kubetorch enable?
# Kubetorch allows you to turn complex operations such as preprocessing and inference into independent services.
# By decoupling accelerated compute tasks from your main application, you can keep the FastAPI app
# light and allows each service to scale independently.
#
# #### Indexing:
# - **Send a list of URL paths** to the application via a POST endpoint
# - Use [LangChain](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)
#   to **parse text from URLs**
# - **Create vector embeddings from split text** with [Sentence Transformers (SBERT)](https://sbert.net/index.html)
# - Store embeddings in a **vector database** ([LanceDB](https://lancedb.com/))
#
# #### Retrieval and generation:
# - **Send question text** via a GET endpoint on the FastAPI application
# - Create a vector embedding from the text and **retrieve related docs** from database
# - **Construct an LLM prompt** from documents and original question
# - Generate a response using an **LLM (Llama 3)**
# - **Output response** with source URLs and question input
#
# ![Graphic displaying the steps of indexing data and the retrieval and generation process](https://runhouse-tutorials.s3.amazonaws.com/indexing-retrieval-generation.png)
#
# Note: Some of the steps in this example could also be accomplished with platforms like OpenAI and
# tools such as LangChain, but we break out the components explicitly to fully illustrate each step and make the
# example easily adaptible to other use cases. Swap out components as you see fit!
#
# ### FastAPI RAG App Setup
# First, we'll import necessary packages and initialize variables used in the application. The `Embedder` and
# `LlamaModel` classes that will be sent to remote compute are available in the `app/modules` folder in this source code.
from typing import Dict, List

import kubetorch as kt

from fastapi import Body, FastAPI, HTTPException

# Template to be used in the LLM generation phase of the RAG app
PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer: """

# ### Create Vector Embedding Service
# This method, run during initialization, will provision remote compute and deploy an embedding service.
#
# The Python packages required by the embedding service (`langchain` etc.) are defined on the image, which
# is a base Docker image and any additional commands to run such as pip installs.
pt_img = kt.Image(image_id="nvcr.io/nvidia/pytorch:25.05-py3").pip_install(
    ["sentence_transformers"]
)


@kt.compute(gpus="1", image=pt_img)
@kt.autoscale(initial_replicas=1, min_replicas=0, max_replicas=4, concurrency=1000)
class Embedder:
    def __init__(self, model_name_or_path="BAAI/bge-large-en-v1.5", **model_kwargs):
        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            model_name_or_path=model_name_or_path, device="cuda", **model_kwargs
        )
        self.model = torch.compile(model)

    def embed(self, text: str, **embed_kwargs):
        return self.model.encode([text], **embed_kwargs)[0].tolist()


# Uncomment to test the Embedder service locally.
# if __name__ == "__main__":
#     Embedder.deploy()
#     res = Embedder.embed("This is a test sentence.", normalize_embeddings=True)
#     print(res)
# Output: [0.12345678, -0.23456789, ...] (example output)


# ### Load RAG LLM Inference Service
# Deploy an open LLM, Llama 3 in this case, to 1 or more GPUs in the cloud.
# We will use vLLM to serve the model due to it's high performance.
#
@kt.compute(
    gpus="1",
    image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.06-py3").run_bash(
        "uv pip install --system --break-system-packages vllm==0.9.0"
    ),
    shared_memory_limit="2Gi",  # Recommended by vLLM: https://docs.vllm.ai/en/v0.6.4/serving/deploying_with_k8s.html
    launch_timeout=1200,  # Need more time to load the model
    secrets=["huggingface"],
)
@kt.autoscale(initial_replicas=1, min_replicas=0, max_replicas=4, concurrency=1000)
class LlamaModel:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        from vllm import LLM

        self.model = LLM(
            model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,  # Reduces size of KV store
            enforce_eager=True,
        )

    def generate(
        self, queries, temperature=0.65, top_p=0.95, max_tokens=5120, min_tokens=32
    ):
        """Generate text with proper error handling and model loading"""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

        req_output = self.model.generate(queries, sampling_params)
        return [output.outputs[0].text for output in req_output]


# ### Initialize LanceDB Vector Database
# We'll be using open source [LanceDB](https://lancedb.com/) to create an embedded database to store
# the URL embeddings and perform vector search for the retrieval phase. You could alternatively try
# Chroma, Pinecone, Weaviate, or even MongoDB.
# We're making these imports optional so we can keep all the service in one file (the other services won't have lance
# in their images), but you can also split this out into a separate file if you prefer.
try:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector

    class Item(LanceModel):
        url: str
        page_content: str
        vector: Vector(1024)

except ImportError:
    pass

lance_lc_image = kt.Image().pip_install(
    [
        "lancedb>=0.3.0",
        "langchain",
        "langchain-community",
        "langchain_text_splitters",
        "langchainhub",
        "bs4",
    ]
)


@kt.compute(cpus=2, memory="8GB", image=lance_lc_image)
class VectorDB:
    def __init__(self):
        self.client = lancedb.connect("/tmp/db")

        self.db = self.client.create_table(
            "rag-table", schema=Item.to_arrow_schema(), exist_ok=True
        )

    def add_document(self, paths: List[str], **embed_kwargs):
        """Generate embeddings for the URL and write to DB."""
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        docs = WebBaseLoader(web_paths=paths).load()
        split_docs = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=50
        ).split_documents(docs)
        splits_as_str = [doc.page_content for doc in split_docs]
        embeddings = Embedder.embed(
            splits_as_str, normalize_embeddings=True, stream_logs=False, **embed_kwargs
        )
        items = [
            Item(
                url=doc.metadata["source"],
                page_content=doc.page_content,
                vector=embeddings[index],
            )
            for index, doc in enumerate(split_docs)
        ]

        self.db.add(items)
        return {"status": "success"}

    def retrieve_documents(self, text: str, limit: int) -> List[Dict]:
        """Retrieve documents from vector DB related to input text"""
        # Encode the input text into a vector
        vector = Embedder.embed(
            text=[text], normalize_embeddings=True, stream_logs=False
        )[0]
        # Search LanceDB for nearest neighbors to the vector embed
        return self.db.search(vector).limit(limit).to_list()


### Initialize the FastAPI Application
# Before defining endpoints, we'll initialize the application and set the lifespan events defined
# above. This will load in the various services we've defined on start-up.
app = FastAPI()

# Add an endpoint to check on our app health. This is a minimal example intended to only
# show if the application is up and running or down.
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ### Vector Embedding POST Endpoint
# To illustrate the flexibility of FastAPI, we're allowing embeddings to be added to your database
# via a POST endpoint. This method will use the embedder service to create database entries with the
# source, content, and vector embeddings for chunks of text from a provided list of URLs.
@app.post("/add")
async def add_document(paths: List[str] = Body([]), kwargs: Dict = Body({})):
    """Generate embeddings for the URL and write to DB."""
    try:
        VectorDB.add_document(
            paths,
            stream_logs=False,
            **kwargs,
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed URLs: {str(e)}")


# ## Retrieval Augmented Generation (RAG) Steps:
# Now that we've defined our services and created an endpoint to populate data for retrieval, the remaining
# components of the application will focus on the generative phases of the RAG app.

# **Retrieval with Sentence Transformers and LanceDB:**
# In the retrieval phase we'll first use the Embedder service to create an embedding from input text
# to search our LanceDB vector database with. LanceDB is optimized for vector searches in this manner.

# **Format an Augmented Prompt**
# To leverage the documents retrieved from the previous step, we'll format a prompt that provides text from
# related documents as "context" for the LLM. This allows a general purpose LLM (like Llama) to provide
# more specific responses to a particular question.
async def format_prompt(text: str, docs: List[Dict]) -> str:
    """Retrieve documents from vector DB related to input text"""
    context = "\n".join([doc["page_content"] for doc in docs])
    prompt = PROMPT_TEMPLATE.format(question=text, context=context)
    return prompt


# ### Generation GET endpoint
# Using the methods above, this endpoint will run inference on our LLM to generate a response to a question.
# The results are enhanced by first retrieving related documents from the source URLs fed into the POST endpoint.
# Content from the fetched documents is then formatted into the text prompt sent to our self-hosted LLM.
# We'll be using a generic prompt template to illustrate how many "chat" tools work behind the scenes.
@app.get("/generate")
async def generate_response(text: str, limit: int = 4):
    """Generate a response to a question using an LLM with context from our database"""
    if not text:
        return {"error": "Question text is missing"}

    try:
        # Retrieve related documents from vector DB
        documents = VectorDB.retrieve_documents(text, limit, stream_logs=False)

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


# ## Run the FastAPI App Locally
# Use the following command to run the app from your terminal:
#
# ```shell
# $ fastapi run app/main.py
# ```
#
# After a few minutes, you can navigate to `http://127.0.0.1/health` to check that your application is running.
# This may take a while due to initialization logic in the lifespan.
#
# You'll see something like:
# ```json
# { "status": "healthy" }
# ```
#
#
# ### Example cURL Command to Add Embeddings
# To populate the LanceDB database with vector embeddings for use in the RAG app, you can send a HTTP request
# to the `/embeddings` POST endpoint. Let's say you have a question about bears. You could send a cURL
# command with a list of URLs including essential bear information:
#
# ```shell
# curl --header "Content-Type: application/json" \
#   --request POST \
#   --data '{"paths":["https://www.nps.gov/yell/planyourvisit/safety.htm", "https://en.wikipedia.org/wiki/Adventures_of_the_Gummi_Bears"]}' \
#   http://127.0.0.1:8000/embeddings
# ```
#
# Alternatively, we recommend a tool like [Postman](https://www.postman.com/) to test HTTP APIs.
#
# ### Test the Generation Endpoint
# Open your browser and send a prompt to your locally running RAG app by appending your question
# to the URL as a query param, e.g. `?text=Does%20yellowstone%20have%20gummi%20bears%3F`
#
# ```text
# "http://127.0.0.1/generate?text=Does%20yellowstone%20have%20gummi%20bears%3F"
# ```
#
# The `LlamaModel` will need to load on the initial call and may take a few minutes to generate a
# response. Subsequent calls will generally take less than a second.
#
# Example output:
#
# ```json
# {
#   "question": "Does yellowstone have gummi bears?",
#   "response": [
#     " No, Yellowstone is bear country, not gummi bear country. Thanks for asking! "
#   ],
#   "sources": [
#     "https://www.nps.gov/yell/planyourvisit/safety.htm",
#     "https://en.wikipedia.org/wiki/Adventures_of_the_Gummi_Bears"
#   ]
# }
# ```
