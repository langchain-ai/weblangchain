"use client";

import { ChatWindow } from "../app/components/ChatWindow";
import { ToastContainer } from "react-toastify";

import { ChakraProvider } from "@chakra-ui/react";

export default function Home() {
  return (
    <ChakraProvider>
      <ToastContainer />
      <ChatWindow
        apiBaseUrl={
          process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8080"
        }
        titleText="WebLangChain 🦜🔗"
        placeholder="Ask anything..."
      ></ChatWindow>
    </ChakraProvider>
  );
}
