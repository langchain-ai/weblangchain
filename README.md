# 🦜️🌐 WebLangChain

This repo is an example of performing retrieval using the entire internet as a document store.

**Try it live:** [weblangchain.vercel.app](https://weblangchain.vercel.app)

## ✅ Running locally

By default, WebLangChain uses [Tavily](https://tavily.com) to fetch content from webpages. You can get an API key from [by signing up](https://tavily.com/).
If you'd like to swap in a different base retriever (e.g. if you want to use your own data source), you can modify the `get_base_retriever()` method in `main.py`.
The code includes a simple backup that uses the Google Custom Search Engine for reference.

1. Install backend dependencies: `poetry install`.
2. Make sure to set your environment variables to configure the application:
```
export OPENAI_API_KEY=
export TAVILY_API_KEY=

# if you'd like to use the backup retriever
export GOOGLE_CSE_ID=
export GOOGLE_API_KEY=

# for tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY=
export LANGCHAIN_PROJECT=
```
3. Start the Python backend with `poetry run make start`.
4. Install frontend dependencies by running `cd nextjs`, then `yarn`.
5. Run the frontend with `yarn dev` for frontend.
6. Open [localhost:3000](http://localhost:3000) in your browser.

## ☕ Running locally (JS backend)

1. Install frontend dependencies by running `cd nextjs`, then `yarn`.
2. Populate a `nextjs/.env.local` file with your own versions of keys from the `nextjs/.env.example` file, and set `NEXT_PUBLIC_API_BASE_URL` to `"http://localhost:3000/api"`.
3. Run the app with `yarn dev`.
4. Open [localhost:3000](http://localhost:3000) in your browser.

## ⚙️ How it works

The general retrieval flow looks like this:

1. Pull in raw content related to the user's initial query using a retriever that wraps Tavily's Search API.
    - For subsequent conversation turns, we also rephrase the original query into a "standalone query" free of references to previous chat history.
2. Because the size of the raw documents usually exceed the maximum context window size of the model, we perform additional [contextual compression steps](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/) to filter what we pass to the model.
    - First, we split retrieved documents using a [text splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/).
    - Then we use an [embeddings filter](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/#embeddingsfilter) to remove any chunks that do not meet a similarity threshold with the initial query.
3. The retrieved context, the chat history, and the original question are passed to the LLM as context for the final generation.

Here's a LangSmith trace illustrating the above:

https://smith.langchain.com/public/f4493d9c-218b-404a-a890-31c15c56fff3/r

It's built using:

- [Tavily](https://tavily.com) as a retriever
- [LangChain](https://github.com/langchain-ai/langchain/) for orchestration
- [LangServe](https://github.com/langchain-ai/langserve) to directly expose LangChain runnables as endpoints
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org) for the frontend

## 🚀 Deployment

The live version is hosted on [Fly.dev](https://fly.dev) and [Vercel](https://vercel.com).
The backend Python logic is found in `main.py`, and the frontend Next.js app is under `nextjs/`.
