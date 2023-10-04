"""Main entrypoint for the app."""
import asyncio
import json
import os
from operator import itemgetter
from typing import AsyncIterator, Dict, List, Optional, Sequence

import langsmith
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline, EmbeddingsFilter)
from langchain.schema import Document
from langchain.schema.document import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (Runnable, RunnableBranch,
                                       RunnableLambda, RunnableMap)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langserve import add_routes
from langsmith import Client
from typing_extensions import TypedDict

RESPONSE_TEMPLATE = """\
You are an expert researcher and writer, tasked with answering any question.

Generate a comprehensive and informative, yet concise answer of 250 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity. If you want to cite multiple results for the same sentence, \
format it as `[${{number1}}] [${{number2}}]`. However, you should NEVER do this with the \
same number - if you want to cite `number1` multiple times for a sentence, only do \
`[${{number1}}]` not `[${{number1}}] [${{number1}}]`

You should use bullet points in your answer for readability. Put citations where they apply \
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context` html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user.

<context>
    {context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

import os
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever


class SearchDepth(Enum):
    BASIC = "basic"
    ADVANCED = "advanced"


class TavilySearchAPIRetriever(BaseRetriever):
    """Tavily Search API retriever."""

    k: int = 10
    include_generated_answer: bool = False
    include_raw_content: bool = False
    include_images: bool = False
    search_depth: SearchDepth = SearchDepth.BASIC
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    kwargs: Optional[Dict[str, Any]] = {}
    api_key: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            from tavily import Client
        except ImportError:
            raise ValueError(
                "Tavily python package not found. "
                "Please install it with `pip install tavily-python`."
            )

        tavily = Client(api_key=self.api_key or os.environ["TAVILY_API_KEY"])
        max_results = self.k if not self.include_generated_answer else self.k - 1
        response = tavily.search(
            query=query,
            max_results=max_results,
            search_depth=self.search_depth.value,
            include_answer=self.include_generated_answer,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            include_raw_content=self.include_raw_content,
            include_images=self.include_images,
            **self.kwargs,
        )
        docs = [
            Document(
                page_content=result.get("content", "")
                if not self.include_raw_content
                else result.get("raw_content", ""),
                metadata={
                    "title": result.get("title", ""),
                    "source": result.get("url", ""),
                    **{
                        k: v
                        for k, v in result.items()
                        if k not in ("content", "title", "url", "raw_content")
                    },
                    "images": response.get("images"),
                },
            )
            for result in response.get("results")
        ]
        if self.include_generated_answer:
            docs = [
                Document(
                    page_content=response.get("answer", ""),
                    metadata={
                        "title": "Suggested Answer",
                        "source": "https://tavily.com/",
                    },
                ),
                *docs,
            ]

        return docs


class ChatRequest(TypedDict):
    question: str
    chat_history: Optional[List[Dict[str, str]]]
    # conversation_id: Optional[str]


def get_base_retriever():
    return TavilySearchAPIRetriever(
        k=6, include_raw_content=True, include_images=True
    )


def _get_retriever():
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    relevance_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, relevance_filter]
    )
    base_retriever = get_base_retriever()
    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    ).with_config(run_name="GetRelevantDocumentChunks")


def create_retriever_chain(
    llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    )


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(llm, retriever) | RunnableLambda(
        format_docs
    ).with_config(run_name="FormatDocumentChunks")
    _context = RunnableMap(
        {
            "context": retriever_chain.with_config(run_name="RetrievalChain"),
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(itemgetter("chat_history")).with_config(
                run_name="Itemgetter:chat_history"
            ),
        }
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
        {
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(serialize_history).with_config(
                run_name="SerializeHistory"
            ),
        }
        | _context
        | response_synthesizer
    )


llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    # model="gpt-4",
    streaming=True,
    temperature=0,
)

retriever = _get_retriever()

chain = create_chain(llm, retriever)

add_routes(app, chain, path="/chat", input_type=ChatRequest)


@app.post("/feedback")
async def send_feedback(request: Request):
    data = await request.json()
    run_id = data.get("run_id")
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    key = data.get("key", "user_score")
    vals = {**data, "key": key}
    client.create_feedback(**vals)
    return {"result": "posted feedback successfully", "code": 200}


@app.patch("/feedback")
async def update_feedback(request: Request):
    data = await request.json()
    feedback_id = data.get("feedback_id")
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=data.get("score"),
        comment=data.get("comment"),
    )
    return {"result": "patched feedback successfully", "code": 200}


# TODO: Update when async API is available
async def _arun(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


async def aget_trace_url(run_id: str) -> str:
    for i in range(5):
        try:
            await _arun(client.read_run, run_id)
            break
        except langsmith.utils.LangSmithError:
            await asyncio.sleep(1**i)

    if await _arun(client.run_is_shared, run_id):
        return await _arun(client.read_run_shared_link, run_id)
    return await _arun(client.share_run, run_id)


@app.post("/get_trace")
async def get_trace(request: Request):
    data = await request.json()
    run_id = data.get("run_id")
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(run_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
