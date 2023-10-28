"use client";

import React, { useRef, useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";

import { v4 as uuidv4 } from "uuid";
import { ChatMessageBubble, Message } from "./ChatMessageBubble";
import { marked } from "marked";
import { Renderer } from "marked";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { applyPatch } from "fast-json-patch";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";

import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Spinner,
  Select,
} from "@chakra-ui/react";
import { ArrowUpIcon } from "@chakra-ui/icons";
import { Source } from "./SourceBubble";
import { DefaultQuestion } from "./DefaultQuestion";

type RetrieverName = "tavily" | "kay" | "you" | "google" | "kay_press_release";

export function ChatWindow(props: {
  apiBaseUrl: string;
  placeholder?: string;
  titleText?: string;
}) {
  const searchParams = useSearchParams();

  const conversationId = uuidv4();
  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<Array<Message>>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [retriever, setRetriever] = useState<RetrieverName>(
    (searchParams.get("retriever") as RetrieverName) ?? "tavily",
  );
  const [llm, setLlm] = useState(searchParams.get("llm") ?? "openai");

  const [chatHistory, setChatHistory] = useState<[string, string][]>([]);

  const { apiBaseUrl, titleText } = props;

  const sendMessage = async (message?: string) => {
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }
    const messageValue = message ?? input;
    if (messageValue === "") return;
    setInput("");
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: Math.random().toString(), content: messageValue, role: "user" },
    ]);
    setIsLoading(true);

    let accumulatedMessage = "";
    let runId: string | undefined = undefined;
    let sources: Source[] | undefined = undefined;
    let messageIndex: number | null = null;

    let renderer = new Renderer();
    renderer.paragraph = (text) => {
      return text + "\n";
    };
    renderer.list = (text) => {
      return `${text}\n\n`;
    };
    renderer.listitem = (text) => {
      return `\n• ${text}`;
    };
    renderer.code = (code, language) => {
      const validLanguage = hljs.getLanguage(language || "")
        ? language
        : "plaintext";
      const highlightedCode = hljs.highlight(
        validLanguage || "plaintext",
        code,
      ).value;
      return `<pre class="highlight bg-gray-700" style="padding: 5px; border-radius: 5px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; max-width: 100%; display: block; line-height: 1.2"><code class="${language}" style="color: #d6e2ef; font-size: 12px; ">${highlightedCode}</code></pre>`;
    };
    marked.setOptions({ renderer });

    try {
      const sourceStepName = "FinalSourceRetriever";
      let streamedResponse: Record<string, any> = {};
      await fetchEventSource(apiBaseUrl + "/chat/stream_log", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          input: {
            question: messageValue,
            chat_history: chatHistory,
          },
          config: {
            configurable: {
              retriever,
              llm,
            },
            metadata: {
              conversation_id: conversationId,
            },
          },
          diff: true,
          include_names: [sourceStepName],
        }),
        openWhenHidden: true,
        onerror(e) {
          throw e;
        },
        onmessage(msg) {
          if (msg.event === "end") {
            setChatHistory((prevChatHistory) => [
              ...prevChatHistory,
              ["human", messageValue],
              ["ai", accumulatedMessage],
            ]);
            setIsLoading(false);
            return;
          }
          if (msg.event === "data" && msg.data) {
            const chunk = JSON.parse(msg.data);
            streamedResponse = applyPatch(
              streamedResponse,
              chunk.ops,
            ).newDocument;
            if (
              Array.isArray(
                streamedResponse?.logs?.[sourceStepName]?.final_output
                  ?.documents,
              )
            ) {
              sources = streamedResponse.logs[
                sourceStepName
              ].final_output.documents.map((doc: Record<string, any>) => ({
                url: doc.metadata.source ?? doc.metadata.data_source_link,
                defaultSourceUrl: retriever === "you" ? "https://you.com" : "",
                title: doc.metadata.title,
                images: doc.metadata.images,
              }));
            }
            if (streamedResponse.id !== undefined) {
              runId = streamedResponse.id;
            }
            if (Array.isArray(streamedResponse?.streamed_output)) {
              accumulatedMessage = streamedResponse.streamed_output.join("");
            }
            const parsedResult = marked.parse(accumulatedMessage);

            setMessages((prevMessages) => {
              let newMessages = [...prevMessages];
              if (messageIndex === null) {
                messageIndex = newMessages.length;
                newMessages.push({
                  id: Math.random().toString(),
                  content: parsedResult.trim(),
                  runId: runId,
                  sources: sources,
                  role: "assistant",
                });
              } else {
                newMessages[messageIndex].content = parsedResult.trim();
                newMessages[messageIndex].runId = runId;
                newMessages[messageIndex].sources = sources;
              }
              return newMessages;
            });
          }
        },
      });
    } catch (e: any) {
      setMessages((prevMessages) => prevMessages.slice(0, -1));
      setIsLoading(false);
      setInput(messageValue);
      toast.error(e.message);
    }
  };

  const defaultQuestions = [
    "what is langchain?",
    "history of mesopotamia",
    "how does a prism separate light",
    "leonardo dicaprio girlfriend",
    "fun gift ideas for software engineers",
    "how do you build a discord bot",
    "what bear is best",
  ];

  const DEFAULT_QUESTIONS: Record<RetrieverName, string[]> = {
    tavily: defaultQuestions,
    you: defaultQuestions,
    google: defaultQuestions,
    kay: [
      "Is Johnson & Johnson increasing its marketing budget?",
      "How is Lululemon adapting to new customer trends?",
      "Which industries are growing in recent 10-Q reports?",
      "Who are Etsy’s competitors?",
      "Which companies reported data breaches?",
      "What were the biggest strategy changes made by Roku in 2023?",
    ],
    kay_press_release: [
      "How is the healthcare industry adopting generative AI tools?",
      "What were the major technological advancements in the renewable energy sector in 2023?",
      "What happened to Intel's acquisition of Tower Semiconductor?",
      "What were the biggest strategy changes made by Roku in 2023?",
    ],
  };

  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
  };

  const insertUrlParam = (key: string, value?: string) => {
    if (window.history.pushState) {
      const searchParams = new URLSearchParams(window.location.search);
      searchParams.set(key, value ?? "");
      const newurl =
        window.location.protocol +
        "//" +
        window.location.host +
        window.location.pathname +
        "?" +
        searchParams.toString();
      window.history.pushState({ path: newurl }, "", newurl);
    }
  };

  return (
    <div
      className={
        "flex flex-col items-center p-8 rounded grow max-h-full h-full" +
        (messages.length === 0 ? " justify-center mb-32" : "")
      }
    >
      <div className="flex flex-col items-center pb-8 w-full">
        <Heading
          fontSize={messages.length > 0 ? "2xl" : "4xl"}
          fontWeight={"medium"}
          mb={1}
          color={"white"}
        >
          {titleText}
        </Heading>
        <Heading
          fontSize={messages.length === 0 ? "xl" : "lg"}
          fontWeight={"normal"}
          mb={1}
          color={"white"}
          marginTop={messages.length === 0 ? "12px" : ""}
        >
          {messages.length > 0
            ? "We appreciate feedback!"
            : "Ask me anything about anything!"}
        </Heading>
        <div className="text-white flex flex-wrap items-center mt-4">
          <div className="flex items-center mb-2">
            <span className="shrink-0 mr-2">Powered by</span>
            <Select
              value={retriever}
              onChange={(e) => {
                insertUrlParam("retriever", e.target.value);
                setRetriever(e.target.value as RetrieverName);
              }}
              width={"212px"}
            >
              <option value="tavily">Tavily</option>
              <option value="kay">Kay.ai SEC Filings</option>
              <option value="kay_press_release">Kay.ai Press Releases</option>
              <option value="you">You.com</option>
              <option value="google">Google</option>
            </Select>
          </div>
          <div className="flex items-center mb-2">
            <span className="shrink-0 ml-2 mr-2">and</span>
            <Select
              value={llm}
              onChange={(e) => {
                insertUrlParam("llm", e.target.value);
                setLlm(e.target.value);
              }}
              width={"212px"}
            >
              <option value="openai">GPT-3.5-Turbo</option>
              <option value="anthropic">Claude-2</option>
              <option value="googlevertex">Google Vertex AI</option>
            </Select>
          </div>
        </div>
      </div>
      <div
        className="flex flex-col-reverse w-full mb-2 overflow-auto"
        ref={messageContainerRef}
      >
        {messages.length > 0
          ? [...messages]
              .reverse()
              .map((m, index) => (
                <ChatMessageBubble
                  key={m.id}
                  message={{ ...m }}
                  aiEmoji="🦜"
                  apiBaseUrl={apiBaseUrl}
                  isMostRecent={index === 0}
                  messageCompleted={!isLoading}
                ></ChatMessageBubble>
              ))
          : ""}
      </div>
      <InputGroup size="md" alignItems={"center"}>
        <Input
          value={input}
          height={"55px"}
          rounded={"full"}
          type={"text"}
          placeholder="Ask anything..."
          textColor={"white"}
          borderColor={"rgb(58, 58, 61)"}
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
        />
        <InputRightElement h="full" paddingRight={"15px"}>
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="Send"
            icon={isLoading ? <Spinner /> : <ArrowUpIcon />}
            type="submit"
            onClick={(e) => {
              e.preventDefault();
              sendMessage();
            }}
          />
        </InputRightElement>
      </InputGroup>
      {messages.length === 0 ? (
        <div className="w-full text-center flex flex-col">
          <div className="flex grow justify-center w-full mt-4 flex-wrap">
            {DEFAULT_QUESTIONS[retriever]
              .slice(0, 4)
              .map((defaultQuestion, i) => {
                return (
                  <DefaultQuestion
                    key={`defaultquestion:${i}`}
                    question={defaultQuestion}
                    onMouseUp={(e) =>
                      sendInitialQuestion(
                        (e.target as HTMLDivElement).innerText,
                      )
                    }
                  ></DefaultQuestion>
                );
              })}
          </div>
          <div className="grow justify-center w-full mt-2 hidden md:flex">
            {DEFAULT_QUESTIONS[retriever].slice(4).map((defaultQuestion, i) => {
              return (
                <DefaultQuestion
                  key={`defaultquestion:${i + 4}`}
                  question={defaultQuestion}
                  onMouseUp={(e) =>
                    sendInitialQuestion((e.target as HTMLDivElement).innerText)
                  }
                ></DefaultQuestion>
              );
            })}
          </div>
        </div>
      ) : (
        ""
      )}

      {messages.length === 0 ? (
        <footer className="flex justify-center absolute bottom-8">
          <a
            href="https://github.com/langchain-ai/weblangchain"
            target="_blank"
            className="text-white flex items-center"
          >
            <img src="/images/github-mark.svg" className="h-4 mr-1" />
            <span>View Source</span>
          </a>
          <a
            href={`${
              process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8080"
            }/chat/playground`}
            target="_blank"
            className="text-white flex items-center ml-8"
          >
            <span>Open Playground</span>
          </a>
        </footer>
      ) : (
        ""
      )}
    </div>
  );
}
