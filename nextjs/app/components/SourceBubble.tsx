import "react-toastify/dist/ReactToastify.css";
import { emojisplosion } from "emojisplosion";

export type Source = {
  url: string;
  title: string;
  images: string[];
  defaultSourceUrl?: string;
};

export function SourceBubble(props: {
  source: Source;
  highlighted: boolean;
  index: number;
  onMouseEnter: () => any;
  onMouseLeave: () => any;
}) {
  const hostname = new URL(
    props.source.url ?? props.source.defaultSourceUrl,
  ).hostname.replace("www.", "");

  return (
    <a
      href={props.source.url ?? props.source.defaultSourceUrl}
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
