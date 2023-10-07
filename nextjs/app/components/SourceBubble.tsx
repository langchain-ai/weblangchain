import "react-toastify/dist/ReactToastify.css";
import { emojisplosion } from "emojisplosion";

export type Source = {
  url: string;
  title: string;
  images: string[];
};

export function SourceBubble(props: {
  source: Source;
  highlighted: boolean;
  index: number;
  onMouseEnter: () => any;
  onMouseLeave: () => any;
}) {
  const cumulativeOffset = function (element: HTMLElement | null) {
    var top = 0,
      left = 0;
    do {
      top += element?.offsetTop || 0;
      left += element?.offsetLeft || 0;
      element = (element?.offsetParent as HTMLElement) || null;
    } while (element);

    return {
      top: top,
      left: left,
    };
  };

  const animateButton = (buttonId: string) => {
    const button = document.getElementById(buttonId);
    button!.classList.add("animate-ping");
    setTimeout(() => {
      button!.classList.remove("animate-ping");
    }, 500);

    emojisplosion({
      emojiCount: 10,
      uniqueness: 1,
      position() {
        const offset = cumulativeOffset(button);

        return {
          x: offset.left + button!.clientWidth / 2,
          y: offset.top + button!.clientHeight / 2,
        };
      },
      emojis: buttonId === "upButton" ? ["👍"] : ["👎"],
    });
  };
  const hostname = new URL(props.source.url).hostname.replace("www.", "");

  return (
    <a
      href={props.source.url}
      target="_blank"
      onMouseEnter={props.onMouseEnter}
      onMouseLeave={props.onMouseLeave}
      className="hover:no-underline"
    >
      <div
        className={`${
          props.highlighted ? "bg-stone-500" : "bg-stone-700"
        } rounded p-4 text-white h-full text-xs flex flex-col mb-4`}
      >
        <div className="line-clamp-4">{props.source.title}</div>
        <div className="text-white mt-auto">
          {hostname} [{props.index}]
        </div>
      </div>
    </a>
  );
}
