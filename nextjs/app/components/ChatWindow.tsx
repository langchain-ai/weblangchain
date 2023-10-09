"use client";

import React, { useRef, useState } from "react";

import { v4 as uuidv4 } from "uuid";
import { EmptyState } from "../components/EmptyState";
import { ChatMessageBubble, Message } from "../components/ChatMessageBubble";
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
  Flex,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Spinner,
} from "@chakra-ui/react";
import { ArrowUpIcon } from "@chakra-ui/icons";
import { Source } from "./SourceBubble";

export function ChatWindow(props: {
  apiBaseUrl: string;
  placeholder?: string;
  titleText?: string;
}) {
  const conversationId = uuidv4();
  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<Array<Message>>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [chatHistory, setChatHistory] = useState<
    { human: string; ai: string }[]
  >([]);

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
            metadata: {
              conversation_id: conversationId,
            },
          },
          include_names: [sourceStepName],
        }),
        onerror(e) {
          throw e;
        },
        onmessage(msg) {
          if (msg.event === "end") {
            setChatHistory((prevChatHistory) => [
              ...prevChatHistory,
              { human: messageValue, ai: accumulatedMessage },
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
                url: doc.metadata.source,
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

  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
  };

  return (
    <div
      className={
        "flex flex-col items-center p-8 rounded grow max-h-full h-full" +
        (messages.length === 0 ? " justify-center mb-32" : "")
      }
    >
      {messages.length > 0 && (
        <Flex direction={"column"} alignItems={"center"} paddingBottom={"20px"}>
          <Heading fontSize="2xl" fontWeight={"medium"} mb={1} color={"white"}>
            {titleText}
          </Heading>
          {/* <Heading fontSize="md" fontWeight={"normal"} mb={1} color={"white"}>
            Powered by{" "}
            <a
              target="_blank"
              href="https://tavily.com"
              className="text-sky-400"
            >
              Tavily
            </a>
          </Heading> */}
          <Heading fontSize="lg" fontWeight={"normal"} mb={1} color={"white"}>
            We appreciate feedback!
          </Heading>
        </Flex>
      )}
      <div
        className="flex flex-col-reverse w-full mb-2 overflow-auto"
        ref={messageContainerRef}
      >
        {messages.length > 0 ? (
          [...messages]
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
        ) : (
          <EmptyState onChoice={sendInitialQuestion} />
        )}
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
          <div className="flex grow justify-center w-full mt-4">
            <div
              onMouseUp={(e) =>
                sendInitialQuestion((e.target as HTMLDivElement).innerText)
              }
              className="bg-stone-700 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-500"
            >
              what is langchain?
            </div>
            <div
              onMouseUp={(e) =>
                sendInitialQuestion((e.target as HTMLDivElement).innerText)
              }
              className="bg-stone-700 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-500"
            >
              history of mesopotamia
            </div>
            <div
              onMouseUp={(e) =>
                sendInitialQuestion((e.target as HTMLDivElement).innerText)
              }
              className="bg-stone-700 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-500"
            >
              how to build a discord bot
            </div>
            <div
              onMouseUp={(e) =>
                sendInitialQuestion((e.target as HTMLDivElement).innerText)
              }
              className="bg-stone-700 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-500"
            >
              leonardo dicaprio girlfriend
            </div>
          </div>
          <div className="flex grow justify-center w-full mt-4">
            <div
              onMouseUp={(e) =>
                sendInitialQuestion((e.target as HTMLDivElement).innerText)
              }
              className="bg-stone-700 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-500"
            >
              fun gift ideas for software engineers
            </div>
            <div
              onMouseUp={(e) =>
                sendInitialQuestion((e.target as HTMLDivElement).innerText)
              }
              className="bg-stone-700 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-500"
            >
              how does a prism separate light
            </div>
            <div
              onMouseUp={(e) =>
                sendInitialQuestion((e.target as HTMLDivElement).innerText)
              }
              className="bg-stone-700 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-500"
            >
              what bear is best
            </div>
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
        </footer>
      ) : (
        ""
      )}
    </div>
  );
}
