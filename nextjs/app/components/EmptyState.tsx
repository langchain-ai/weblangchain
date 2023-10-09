import { Heading } from "@chakra-ui/react";

export function EmptyState(props: { onChoice: (question: string) => any }) {
  return (
    <div className="rounded flex flex-col items-center max-w-full md:p-8">
      <Heading fontSize="3xl" fontWeight={"medium"} mb={1} color={"white"}>
        WebLangChain 🦜🔗
      </Heading>
      {/* <Heading fontSize="md" fontWeight={"normal"} mb={1} color={"white"}>
        Powered by{" "}
        <a target="_blank" href="https://tavily.com" className="text-sky-400">
          Tavily
        </a>
      </Heading> */}
      <Heading
        fontSize="xl"
        fontWeight={"normal"}
        mb={1}
        color={"white"}
        marginTop={"10px"}
        textAlign={"center"}
      >
        Ask me anything about anything!{" "}
      </Heading>
    </div>
  );
}
