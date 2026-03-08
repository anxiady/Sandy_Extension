import dotenv from "dotenv";
import { Message } from "../../type";
import {
  ChatWithLLMStreamFunction,
  SummaryTextWithLLMFunction,
} from "../interface";

dotenv.config();

const whisplayBridgeHost = process.env.WHISPLAY_IM_BRIDGE_HOST || "127.0.0.1";
const whisplayBridgePort = parseInt(
  process.env.WHISPLAY_IM_BRIDGE_PORT || "18888",
);
const whisplayInboxPath =
  process.env.WHISPLAY_IM_INBOX_PATH || "/whisplay-im/inbox";
const whisplayToken = process.env.WHISPLAY_IM_TOKEN || "";
const whisplayTimeoutMs = parseInt(
  process.env.WHISPLAY_IM_TIMEOUT_MS || "30000",
);

const whisplayInboxUrl = `http://${whisplayBridgeHost}:${whisplayBridgePort}${whisplayInboxPath}`;

const resetChatHistory = (): void => {};

export const sendWhisplayIMMessage = async (
  inputMessages: Message[] = [],
): Promise<boolean> => {
  const lastUserMessage = [...inputMessages]
    .reverse()
    .find((msg) => msg.role === "user");

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), whisplayTimeoutMs);

  try {
    const response = await fetch(whisplayInboxUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(whisplayToken ? { Authorization: `Bearer ${whisplayToken}` } : {}),
      },
      body: JSON.stringify({
        message: lastUserMessage?.content || "",
        messages: inputMessages,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.error(`Whisplay IM HTTP error: ${response.status}`);
      return false;
    }

    return true;
  } catch (error: any) {
    if (error?.name === "AbortError") {
      console.error("Whisplay IM request timeout.");
    } else {
      console.error("Whisplay IM request failed:", error);
    }
    return false;
  } finally {
    clearTimeout(timeout);
  }
};

const chatWithLLMStream: ChatWithLLMStreamFunction = async (
  inputMessages: Message[] = [],
  _partialCallback: (partialAnswer: string) => void,
  endCallback: () => void,
): Promise<void> => {
  await sendWhisplayIMMessage(inputMessages);
  endCallback();
};

const summaryTextWithLLM: SummaryTextWithLLMFunction = async (
  text: string,
): Promise<string> => text;

export default { chatWithLLMStream, resetChatHistory, summaryTextWithLLM };
