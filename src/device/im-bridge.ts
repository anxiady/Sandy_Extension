import { EventEmitter } from "events";
import http, { IncomingMessage, ServerResponse } from "http";
import dotenv from "dotenv";

dotenv.config();

type WhisplayIMPayload = {
  message?: string;
  messages?: Array<{ role: string; content: string }>;
  emoji?: string;
};

type PendingPoll = {
  res: ServerResponse;
  timer: NodeJS.Timeout;
};

export class WhisplayIMBridgeServer extends EventEmitter {
  private server: http.Server | null = null;
  private port: number;
  private token: string;
  private inboxPath: string;
  private pollPath: string;
  private sendPath: string;
  private queue: WhisplayIMPayload[] = [];
  private pending: PendingPoll[] = [];

  constructor() {
    super();
    this.port = parseInt(process.env.WHISPLAY_IM_BRIDGE_PORT || "18888");
    this.inboxPath = process.env.WHISPLAY_IM_INBOX_PATH || "/whisplay-im/inbox";
    this.pollPath = process.env.WHISPLAY_IM_POLL_PATH || "/whisplay-im/poll";
    this.sendPath = process.env.WHISPLAY_IM_SEND_PATH || "/whisplay-im/send";
    this.token = process.env.WHISPLAY_IM_TOKEN || "";
  }

  start(): void {
    if (this.server) return;

    this.server = http.createServer((req: IncomingMessage, res: ServerResponse) => {
      const requestUrl = new URL(req.url || "", `http://localhost:${this.port}`);
      const pathname = requestUrl.pathname;

      if (this.token) {
        const auth = req.headers.authorization || "";
        if (auth !== `Bearer ${this.token}`) {
          res.statusCode = 401;
          res.end("Unauthorized");
          return;
        }
      }

      if (req.method === "GET" && pathname === this.pollPath) {
        const waitSec = parseInt(requestUrl.searchParams.get("waitSec") || "0");
        this.handlePoll(res, waitSec);
        return;
      }

      if (req.method !== "POST") {
        res.statusCode = 404;
        res.end("Not Found");
        return;
      }

      let body = "";
      req.on("data", (chunk) => {
        body += chunk.toString();
      });
      req.on("end", () => {
        try {
          const payload = JSON.parse(body || "{}") as WhisplayIMPayload & {
            reply?: string;
          };
          if (pathname === this.inboxPath) {
            this.enqueue(payload);
            res.statusCode = 200;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true }));
            return;
          }
          if (pathname === this.sendPath) {
            const reply = payload.reply || payload.message || "";
            if (reply) {
              this.emit("reply", { reply, emoji: payload.emoji || "" });
            }
            res.statusCode = 200;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true }));
            return;
          }

          res.statusCode = 404;
          res.end("Not Found");
        } catch (error) {
          res.statusCode = 400;
          res.end("Bad Request");
        }
      });
    });

    this.server.listen(this.port, () => {
      console.log(
        `[WhisplayIM] Bridge server listening on ${this.port}${this.inboxPath}`,
      );
    });
  }

  stop(): void {
    if (!this.server) return;
    this.server.close();
    this.server = null;
  }

  private enqueue(payload: WhisplayIMPayload): void {
    if (this.pending.length > 0) {
      const pending = this.pending.shift();
      if (pending) {
        clearTimeout(pending.timer);
        this.respondWithMessage(pending.res, payload);
      }
      return;
    }
    this.queue.push(payload);
  }

  private handlePoll(res: ServerResponse, waitSec: number): void {
    if (this.queue.length > 0) {
      const payload = this.queue.shift();
      this.respondWithMessage(res, payload || {});
      return;
    }

    if (waitSec <= 0) {
      this.respondWithMessage(res, {});
      return;
    }

    const timer = setTimeout(() => {
      this.pending = this.pending.filter((item) => item.res !== res);
      this.respondWithMessage(res, {});
    }, waitSec * 1000);

    this.pending.push({ res, timer });
  }

  private respondWithMessage(res: ServerResponse, payload: WhisplayIMPayload): void {
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ message: payload.message || "", messages: payload.messages || [] }));
  }
}
