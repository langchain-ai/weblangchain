"""Main entrypoint for the app."""
import asyncio
import os
from operator import itemgetter
from typing import List, Optional, Sequence, Tuple, Union

import langsmith
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chat_models import ChatAnthropic, ChatOpenAI, ChatVertexAI
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.retrievers import (
    ContextualCompressionRetriever,
    TavilySearchAPIRetriever,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.kay import KayAiRetriever
from langchain.retrievers.you import YouRetriever
from langchain.schema import Document
from langchain.schema.document import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Backup
from langchain.utilities import GoogleSearchAPIWrapper
from langserve import add_routes
from langsmith import Client
from pydantic import BaseModel, Field
from uuid import UUID

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

Anything between the following `context` html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user.

<context>
    {context}
<context/>

REMEMBER: Anything between the preceding 'context' \
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


class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question", "output": "answer"}},
    )


class GoogleCustomSearchRetriever(BaseRetriever):
    search: Optional[GoogleSearchAPIWrapper] = None
    num_search_results = 6

    def clean_search_query(self, query: str) -> str:
        # Some search tools (e.g., Google) will
        # fail to return results if query has a
        # leading digit: 1. "LangCh..."
        # Check if the first character is a digit
        if query[0].isdigit():
            # Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                # Extract the part of the string after the quote
                query = query[first_quote_pos + 1 :]
                # Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()

    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]:
        """Returns num_search_results pages per Google search."""
        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean, num_search_results)
        return result

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        if os.environ.get("GOOGLE_API_KEY", None) == None:
            raise Exception("No Google API key provided")

        if self.search == None:
            self.search = GoogleSearchAPIWrapper()

        # Get search questions
        print("Generating questions for Google Search ...")

        # Get urls
        print("Searching for relevant urls...")
        urls_to_look = []
        search_results = self.search_tool(query, self.num_search_results)
        print("Searching for relevant urls...")
        print(f"Search results: {search_results}")
        for res in search_results:
            if res.get("link", None):
                urls_to_look.append(res["link"])

        print(search_results)
        loader = AsyncHtmlLoader(urls_to_look)
        html2text = Html2TextTransformer()
        print("Indexing new urls...")
        docs = loader.load()
        docs = list(html2text.transform_documents(docs))
        for i in range(len(docs)):
            if search_results[i].get("title", None):
                docs[i].metadata["title"] = search_results[i]["title"]
        return docs


def get_retriever():
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    relevance_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, relevance_filter]
    )
    base_tavily_retriever = TavilySearchAPIRetriever(
        k=6, include_raw_content=True, include_images=True
    )
    tavily_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_tavily_retriever
    )
    base_google_retriever = GoogleCustomSearchRetriever()
    google_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_google_retriever
    )
    base_you_retriever = YouRetriever(
        ydc_api_key=os.environ.get("YDC_API_KEY", "not_provided")
    )
    you_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_you_retriever
    )
    base_kay_retriever = KayAiRetriever.create(
        dataset_id="company",
        data_types=["10-K", "10-Q"],
        num_contexts=6,
    )
    kay_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_kay_retriever
    )
    base_kay_press_release_retriever = KayAiRetriever.create(
        dataset_id="company",
        data_types=["PressRelease"],
        num_contexts=6,
    )
    kay_press_release_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_kay_press_release_retriever,
    )
    return tavily_retriever.configurable_alternatives(
        # This gives this field an id
        # When configuring the end runnable, we can then use this id to configure this field
        ConfigurableField(id="retriever"),
        default_key="tavily",
        google=google_retriever,
        you=you_retriever,
        kay=kay_retriever,
        kay_press_release=kay_press_release_retriever,
    ).with_config(run_name="FinalSourceRetriever")


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
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def serialize_history(request: ChatRequest):
    chat_history = request.get("chat_history", [])
    converted_chat_history = []
    for message in chat_history:
        if message[0] == "human":
            converted_chat_history.append(HumanMessage(content=message[1]))
        elif message[0] == "ai":
            converted_chat_history.append(AIMessage(content=message[1]))
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


dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    dir_path + "/" + ".google_vertex_ai_credentials.json"
)

has_google_creds = os.path.isfile(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])

llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    # model="gpt-4",
    streaming=True,
    temperature=0.1,
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    default_key="openai",
    anthropic=ChatAnthropic(
        model="claude-2",
        max_tokens=16384,
        temperature=0.1,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "not_provided"),
    ),
)

if has_google_creds:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        # model="gpt-4",
        streaming=True,
        temperature=0.1,
    ).configurable_alternatives(
        # This gives this field an id
        # When configuring the end runnable, we can then use this id to configure this field
        ConfigurableField(id="llm"),
        default_key="openai",
        anthropic=ChatAnthropic(
            model="claude-2",
            max_tokens=16384,
            temperature=0.1,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "not_provided"),
        ),
        googlevertex=ChatVertexAI(
            model_name="chat-bison-32k",
            temperature=0.1,
            max_output_tokens=8192,
            stream=True,
        ),
    )

retriever = get_retriever()

chain = create_chain(llm, retriever)

add_routes(
    app, chain, path="/chat", input_type=ChatRequest, config_keys=["configurable"]
)


class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


@app.post("/feedback")
async def send_feedback(body: SendFeedbackBody):
    client.create_feedback(
        body.run_id,
        body.key,
        score=body.score,
        comment=body.comment,
        feedback_id=body.feedback_id,
    )
    return {"result": "posted feedback successfully", "code": 200}


class UpdateFeedbackBody(BaseModel):
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None


@app.patch("/feedback")
async def update_feedback(body: UpdateFeedbackBody):
    feedback_id = body.feedback_id
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=body.score,
        comment=body.comment,
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


class GetTraceBody(BaseModel):
    run_id: UUID


@app.post("/get_trace")
async def get_trace(body: GetTraceBody):
    run_id = body.run_id
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(str(run_id))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
