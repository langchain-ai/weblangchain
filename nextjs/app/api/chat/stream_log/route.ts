// JS backend not used by default, see README for instructions.

import { NextRequest, NextResponse } from "next/server";

import type { BaseLanguageModel } from "langchain/base_language";
import type { Document } from "langchain/document";

import {
  RunnableSequence,
  RunnableMap,
  RunnableBranch,
  RunnableLambda,
  Runnable,
} from "langchain/schema/runnable";
import { HumanMessage, AIMessage, BaseMessage } from "langchain/schema";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { StringOutputParser } from "langchain/schema/output_parser";
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "langchain/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { TavilySearchAPIRetriever } from "langchain/retrievers/tavily_search_api";
import { DocumentCompressorPipeline } from "langchain/retrievers/document_compressors";
import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";
import { EmbeddingsFilter } from "langchain/retrievers/document_compressors/embeddings_filter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

export const runtime = "edge";

const RESPONSE_TEMPLATE = `You are an expert researcher and writer, tasked with answering any question.

Generate a comprehensive and informative, yet concise answer of 250 words or less for the
given question based solely on the provided search results (URL and content). You must
only use information from the provided search results. Use an unbiased and
journalistic tone. Combine search results together into a coherent answer. Do not
repeat text. Cite search results using [\${{number}}] notation. Only cite the most
relevant results that answer the question accurately. Place these citations at the end
of the sentence or paragraph that reference them - do not put them all at the end. If
different results refer to different entities within the same name, write separate
answers for each entity. If you want to cite multiple results for the same sentence,
format it as \`[\${{number1}}] [\${{number2}}]\`. However, you should NEVER do this with the
same number - if you want to cite \`number1\` multiple times for a sentence, only do
\`[\${{number1}}]\` not \`[\${{number1}}] [\${{number1}}]\`

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

Anything between the following \`context\` html blocks is retrieved from a knowledge
bank, not part of the conversation with the user.

<context>
    {context}
<context/>

REMEMBER: Anything between the preceding 'context'
html blocks is retrieved from a knowledge bank, not part of the conversation with the
user.`;

const REPHRASE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:`;

type RetrievalChainInput = {
  chat_history: string;
  question: string;
};

/**
 * Override with your own retriever if desired.
 */
const getBaseRetriever = () => {
  return new TavilySearchAPIRetriever({
    k: 6,
    includeRawContent: true,
    includeImages: true,
  });
};

const _getRetriever = () => {
  const embeddings = new OpenAIEmbeddings({});
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 800,
    chunkOverlap: 20,
  });
  const relevanceFilter = new EmbeddingsFilter({
    embeddings,
    similarityThreshold: 0.8,
  });
  const pipelineCompressor = new DocumentCompressorPipeline({
    transformers: [splitter, relevanceFilter],
  });
  const baseRetriever = getBaseRetriever();
  return new ContextualCompressionRetriever({
    baseCompressor: pipelineCompressor,
    baseRetriever,
  }).withConfig({
    runName: "FinalSourceRetriever",
  });
};

const createRetrieverChain = (llm: BaseLanguageModel, retriever: Runnable) => {
  const CONDENSE_QUESTION_PROMPT =
    PromptTemplate.fromTemplate(REPHRASE_TEMPLATE);
  const condenseQuestionChain = RunnableSequence.from([
    CONDENSE_QUESTION_PROMPT,
    llm,
    new StringOutputParser(),
  ]).withConfig({
    runName: "CondenseQuestion",
  });
  const hasHistoryCheckFn = RunnableLambda.from(
    (input: RetrievalChainInput) => input.chat_history.length > 0,
  ).withConfig({ runName: "HasChatHistoryCheck" });
  const conversationChain = condenseQuestionChain.pipe(retriever).withConfig({
    runName: "RetrievalChainWithHistory",
  });
  const basicRetrievalChain = RunnableLambda.from(
    (input: RetrievalChainInput) => input.question,
  )
    .withConfig({
      runName: "Itemgetter:question",
    })
    .pipe(retriever)
    .withConfig({ runName: "RetrievalChainWithNoHistory" });

  return RunnableBranch.from([
    [hasHistoryCheckFn, conversationChain],
    basicRetrievalChain,
  ]).withConfig({
    runName: "RouteDependingOnChatHistory",
  });
};

const serializeHistory = (input: any) => {
  const chatHistory = input.chat_history || [];
  const convertedChatHistory = [];
  for (const message of chatHistory) {
    if (message[0] === "human") {
      convertedChatHistory.push(new HumanMessage({ content: message[1] }));
    } else if (message[0] === "ai") {
      convertedChatHistory.push(new AIMessage({ content: message[1] }));
    }
  }
  return convertedChatHistory;
};

const formatDocs = (docs: Document[]) => {
  return docs
    .map((doc, i) => `<doc id='${i}'>${doc.pageContent}</doc>`)
    .join("\n");
};

const formatChatHistoryAsString = (history: BaseMessage[]) => {
  return history
    .map((message) => `${message._getType()}: ${message.content}`)
    .join("\n");
};

const createChain = (llm: BaseLanguageModel, retriever: Runnable) => {
  const retrieverChain = createRetrieverChain(llm, retriever).pipe(
    RunnableLambda.from(formatDocs).withConfig({
      runName: "FormatDocumentChunks",
    }),
  );
  const context = RunnableMap.from({
    context: RunnableSequence.from([
      ({ question, chat_history }) => ({
        question,
        chat_history: formatChatHistoryAsString(chat_history),
      }),
      retrieverChain,
    ]),
    question: RunnableLambda.from(
      (input: RetrievalChainInput) => input.question,
    ).withConfig({
      runName: "Itemgetter:question",
    }),
    chat_history: RunnableLambda.from(
      (input: RetrievalChainInput) => input.chat_history,
    ).withConfig({
      runName: "Itemgetter:chat_history",
    }),
  }).withConfig({ tags: ["RetrieveDocs"] });
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", RESPONSE_TEMPLATE],
    new MessagesPlaceholder("chat_history"),
    ["human", "{question}"],
  ]);

  const responseSynthesizerChain = RunnableSequence.from([
    prompt,
    llm,
    new StringOutputParser(),
  ]).withConfig({
    tags: ["GenerateResponse"],
  });
  return RunnableSequence.from([
    {
      question: RunnableLambda.from(
        (input: RetrievalChainInput) => input.question,
      ).withConfig({
        runName: "Itemgetter:question",
      }),
      chat_history: RunnableLambda.from(serializeHistory).withConfig({
        runName: "SerializeHistory",
      }),
    },
    context,
    responseSynthesizerChain,
  ]);
};

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const input = body.input;
    const config = body.config;
    const llm = new ChatOpenAI({
      modelName: "gpt-3.5-turbo-16k",
      temperature: 0,
    });
    const retriever = await _getRetriever();
    const answerChain = createChain(llm, retriever);

    // Narrows streamed log output down to final output and the FindDocs tagged chain to
    // selectively stream back sources.
    const stream = await answerChain.streamLog(input, config, {
      includeNames: body.include_names,
    });

    const encoder = new TextEncoder();
    const finalStream = new ReadableStream({
      async start(controller) {
        for await (const chunk of stream) {
          controller.enqueue(
            encoder.encode(
              "event: data\ndata: " + JSON.stringify(chunk) + "\n\n",
            ),
          );
        }
        controller.enqueue(encoder.encode("event: end\n\n"));
        controller.close();
      },
    });

    return new Response(finalStream, {
      headers: {
        "content-type": "text/event-stream",
      },
    });
  } catch (e: any) {
    console.log(e);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
