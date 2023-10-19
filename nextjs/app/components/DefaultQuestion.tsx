import { MouseEventHandler } from "react";

export function DefaultQuestion(props: {
  question: string;
  onMouseUp: MouseEventHandler;
}) {
  return (
    <div
      onMouseUp={props.onMouseUp}
      className="bg-stone-700 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-500 mb-2"
    >
      {props.question}
    </div>
  );
}
