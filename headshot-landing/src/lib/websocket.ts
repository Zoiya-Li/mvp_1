"use client";

import { useEffect, useRef, useState } from "react";
import type { WSMessage } from "./types";
import { getSessionToken } from "./api";

const MAX_MESSAGES = 200; // cap so a long session doesn't grow memory unbounded
const MAX_RETRIES = 5;

export type WSConnectionStatus =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected";

/**
 * Real-time generation updates for a session.
 *
 * - Authenticates the handshake with the session's owner token (?token=).
 * - Reconnects with exponential backoff on unexpected close (a flaky network
 *   shouldn't silently freeze the progress UI). Gives up after MAX_RETRIES.
 * - Caps the buffered messages to MAX_MESSAGES.
 * - Delivers each message via ``onMessage`` (a ref-stable callback) rather than
 *   exposing latestMessage — this keeps message handling off the render→effect
 *   cascade and is the React-recommended way to bridge an external stream.
 */
export function useWebSocket(
  sessionId: string | null,
  onMessage?: (msg: WSMessage) => void
) {
  const [messages, setMessages] = useState<WSMessage[]>([]);
  const [status, setStatus] = useState<WSConnectionStatus>("disconnected");

  // Keep the latest onMessage in a ref so ws.onmessage (created once) always
  // calls the current closure without rebuilding the socket. Synced in an effect
  // (not during render) — mutating refs during render is flagged by the
  // react-hooks rules and can cause skipped updates.
  const onMessageRef = useRef(onMessage);
  useEffect(() => {
    onMessageRef.current = onMessage;
  });

  useEffect(() => {
    // No active session → no connection. We derive the reported status as
    // "disconnected" in that case (see returned `status`), so there is no need
    // to call setState here synchronously (which would trip the
    // set-state-in-effect rule).
    if (!sessionId) return;

    // `sessionId` is a nullable param; narrow it to a const string so the
    // narrowing holds inside the nested `connect()` closure (TS does not carry
    // param narrowing into nested function declarations).
    const sid = sessionId;

    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let retry = 0;
    let manualClose = false;

    // Function declaration (hoisted) so onclose can reference it for reconnect.
    function connect() {
      const token = getSessionToken(sid);
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const tokenQuery = token ? `?token=${encodeURIComponent(token)}` : "";
      const url = `${protocol}//${window.location.host}/ws/${sid}${tokenQuery}`;

      setStatus(retry === 0 ? "connecting" : "reconnecting");
      const ws = new WebSocket(url);
      socket = ws;

      ws.onopen = () => {
        retry = 0;
        setStatus("connected");
      };

      ws.onclose = () => {
        socket = null;
        if (manualClose) {
          setStatus("disconnected");
          return;
        }
        if (retry < MAX_RETRIES) {
          const attempt = retry;
          retry += 1;
          const backoff = Math.min(1000 * 2 ** attempt, 16_000);
          setStatus("reconnecting");
          reconnectTimer = setTimeout(connect, backoff);
        } else {
          setStatus("disconnected");
        }
      };

      ws.onerror = () => {
        // onclose handles the reconnect strategy; error alone has no policy.
        setStatus("reconnecting");
      };

      ws.onmessage = (event) => {
        try {
          const msg: WSMessage = JSON.parse(event.data);
          setMessages((prev) => {
            const next = [...prev, msg];
            return next.length > MAX_MESSAGES ? next.slice(-MAX_MESSAGES) : next;
          });
          onMessageRef.current?.(msg);
        } catch {
          /* ignore malformed frames */
        }
      };
    }

    connect();

    // Keep-alive ping every 30s.
    const ping = setInterval(() => {
      if (socket?.readyState === WebSocket.OPEN) socket.send("ping");
    }, 30_000);

    return () => {
      manualClose = true;
      clearInterval(ping);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
      socket = null;
    };
  }, [sessionId]);

  const latestMessage = messages[messages.length - 1] ?? null;

  // When there's no session, always report disconnected regardless of the
  // (possibly stale) internal status — keeps a frozen "reconnecting" from
  // lingering after the session ends.
  const reportedStatus = sessionId ? status : "disconnected";

  return { messages, latestMessage, status: reportedStatus };
}
