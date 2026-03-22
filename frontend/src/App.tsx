"use client";

import React, { useState, useEffect } from "react";
import { Send, User, Loader2, Save } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

type Role = "user" | "assistant";

type RetrievedItem = {
  source: string;
  id?: string | null;
  content: string;
  score?: number | null;
  timestamp?: string | null;
  topic?: string | null;
  topic_summary?: string | null;
};

type NoteWriteItem = {
  id?: string | null;
  memory?: string | null;
  category?: string | null;
  event?: string | null;
};

type EpisodeWriteItem = {
  topic?: string | null;
  topic_summary?: string | null;
  document_id?: string | null;
};

type MemoryTrace = {
  query: string;
  retrieved: RetrievedItem[];
  note_writes: NoteWriteItem[];
  episode_writes: EpisodeWriteItem[];
};

type ChatTurn = {
  id: number;
  role: Role;
  text: string;
  timestamp?: string;
  memoryTrace?: MemoryTrace;
};

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

function MemoryThisTurn({ trace }: { trace: MemoryTrace }) {
  const hasRetrieved = trace.retrieved.length > 0;
  const hasNotes = trace.note_writes.length > 0;
  const hasEpisodes = trace.episode_writes.length > 0;

  return (
    <details className="border-border bg-muted/40 mt-2 max-w-full rounded-lg border text-left">
      <summary className="text-muted-foreground cursor-pointer select-none px-3 py-2 text-xs font-medium">
        Memory this turn
      </summary>
      <div className="text-muted-foreground space-y-3 border-t px-3 py-2 text-xs">
        <p className="text-[11px] opacity-80">
          <span className="font-medium text-foreground/80">Query used for search:</span> {trace.query || "—"}
        </p>
        <div>
          <p className="text-foreground/90 mb-1 font-medium">Retrieved</p>
          {!hasRetrieved ? (
            <p>None</p>
          ) : (
            <ul className="list-inside list-disc space-y-1 font-mono text-[11px] leading-relaxed">
              {trace.retrieved.map((r, i) => (
                <li key={`${r.id ?? i}-${i}`}>
                  <span className="text-foreground/80">[{r.source}{r.timestamp ? ` · ${r.timestamp}` : ""}]</span>{" "}
                  {r.score != null
                    ? `(score ${typeof r.score === "number" ? r.score.toFixed(3) : r.score}) `
                    : ""}
                  {r.content.slice(0, 280)}
                  {r.content.length > 280 ? "…" : ""}
                  {r.topic ? ` · topic: ${r.topic}` : ""}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <p className="text-foreground/90 mb-1 font-medium">Note memory (writes)</p>
          {!hasNotes ? (
            <p>None</p>
          ) : (
            <ul className="list-inside list-disc space-y-1 font-mono text-[11px] leading-relaxed">
              {trace.note_writes.map((n, i) => (
                <li key={`${n.id ?? i}-${i}`}>
                  <span className="text-foreground/80">[{n.event ?? "?"}]</span>{" "}
                  {n.category ? `${n.category}: ` : ""}
                  {n.memory ?? ""}
                  {n.id ? ` · id: ${n.id}` : ""}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <p className="text-foreground/90 mb-1 font-medium">Episodic memory (writes)</p>
          {!hasEpisodes ? (
            <p>None</p>
          ) : (
            <ul className="list-inside list-disc space-y-1 font-mono text-[11px] leading-relaxed">
              {trace.episode_writes.map((e, i) => (
                <li key={`${e.document_id ?? i}-${i}`}>
                  {e.topic ?? "(no topic)"}
                  {e.topic_summary ? ` — ${e.topic_summary}` : ""}
                  {e.document_id != null ? ` · doc: ${e.document_id}` : " · doc: (skipped / unavailable)"}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </details>
  );
}

export default function FullscreenChatInterface() {
  const [messages, setMessages] = useState<ChatTurn[]>([
    {
      id: 1,
      role: "assistant",
      text: "m ira hu app kaise ho",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [sending, setSending] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [token, setToken] = useState<string | null>(localStorage.getItem("active_token"));
  const [isLoginView, setIsLoginView] = useState(true);
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authConfirmPassword, setAuthConfirmPassword] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;
    
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/chat/history`, {
          headers: { "Authorization": `Bearer ${token}` }
        });
        if (res.ok) {
          const data = await res.json();
          if (data && data.length > 0) {
            setMessages([{ id: 1, role: "assistant", text: "m ira hu app kaise ho" }, ...data]);
          } else {
            setMessages([{ id: Date.now(), role: "assistant", text: "m ira hu app kaise ho" }]);
          }
        }
      } catch (e) {
        console.error("Failed to load history", e);
      }
    };
    
    fetchHistory();
  }, [token]);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!authUsername.trim() || !authPassword.trim()) return;
    
    if (!isLoginView && authPassword !== authConfirmPassword) {
      setAuthError("Passwords do not match");
      return;
    }

    setAuthError(null);
    const endpoint = isLoginView ? "/api/auth/login" : "/api/auth/register";
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: authUsername.trim(), password: authPassword }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Authentication Failed");
      
      if (!isLoginView) {
        setIsLoginView(true);
        setAuthError("Registration successful! Please log in.");
        return;
      }
      
      localStorage.setItem("active_token", data.access_token);
      setToken(data.access_token);
      setAuthPassword("");
    } catch (err: any) {
      setAuthError(err.message);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("active_token");
    setToken(null);
    setMessages([{ id: Date.now(), role: "assistant", text: "m ira hu app kaise ho" }]);
  };

  const handleSaveChat = async () => {
    setIsSaving(true);
    try {
      const res = await fetch(`${API_BASE}/api/chat/save`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setMessages((prev) => [...prev, { 
          id: Date.now(), 
          role: "assistant", 
          text: "Chat successfully packed and saved to your long-term memory! Let's start a fresh episode.",
          memoryTrace: data.memory_trace ? {
            query: data.memory_trace.query ?? "",
            retrieved: data.memory_trace.retrieved ?? [],
            note_writes: data.memory_trace.note_writes ?? [],
            episode_writes: data.memory_trace.episode_writes ?? []
          } : undefined
        }]);
      }
    } catch (e) {
      console.error("Failed to save session", e);
    } finally {
      setIsSaving(false);
    }
  };

  const handleSendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const trimmed = inputValue.trim();
    if (!trimmed || sending) return;

    const userTurn: ChatTurn = {
      id: Date.now(),
      role: "user",
      text: trimmed,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userTurn]);
    setInputValue("");
    setError(null);
    setSending(true);

    const payload = {
      user_id: "default",
      messages: [...messages, userTurn].map((m) => ({
        role: m.role,
        content: m.text,
      })),
    };

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify(payload),
      });
      const data = (await res.json().catch(() => ({}))) as {
        detail?: string | unknown;
        message?: { content?: string; timestamp?: string };
        memory_trace?: MemoryTrace;
      };

      if (!res.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
              ? JSON.stringify(data.detail)
              : res.statusText;
        throw new Error(detail || `Request failed (${res.status})`);
      }

      const reply = data.message?.content?.trim();
      if (!reply) throw new Error("Empty response from server");

      const trace = data.memory_trace;

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: reply,
          timestamp: data.message?.timestamp,
          memoryTrace: trace
            ? {
                query: trace.query ?? "",
                retrieved: trace.retrieved ?? [],
                note_writes: trace.note_writes ?? [],
                episode_writes: trace.episode_writes ?? [],
              }
            : undefined,
        },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Something went wrong";
      setError(msg);
    } finally {
      setSending(false);
    }
  };

  if (!token) {
    return (
      <div className="flex h-screen w-full flex-col items-center justify-center bg-slate-50">
        <div className="w-full max-w-sm rounded-xl border bg-white p-6 shadow-sm">
          <div className="mb-6 flex flex-col items-center gap-2">
            <Avatar className="h-12 w-12">
              <AvatarFallback className="bg-primary text-primary-foreground font-heading italic">
                <span className="italic">i</span>
              </AvatarFallback>
            </Avatar>
            <h1 className="text-2xl font-semibold">ira</h1>
            <p className="text-muted-foreground text-sm">{isLoginView ? "Log in to your account" : "Create an account"}</p>
          </div>
          <form onSubmit={handleAuth} className="flex flex-col gap-4">
            {authError && <p className="text-sm font-medium text-destructive text-center">{authError}</p>}
            <Input
              placeholder="Username"
              value={authUsername}
              onChange={(e) => setAuthUsername(e.target.value)}
              disabled={sending}
            />
            <Input
              type="password"
              placeholder="Password"
              value={authPassword}
              onChange={(e) => setAuthPassword(e.target.value)}
              disabled={sending}
            />
            {!isLoginView && (
              <Input
                type="password"
                placeholder="Retype Password"
                value={authConfirmPassword}
                onChange={(e) => setAuthConfirmPassword(e.target.value)}
                disabled={sending}
              />
            )}
            <Button type="submit" className="w-full">
              {isLoginView ? "Log In" : "Register"}
            </Button>
          </form>
          <div className="mt-4 text-center text-sm">
            <button
              onClick={() => { 
                setIsLoginView(!isLoginView); 
                setAuthError(null);
                setAuthPassword("");
                setAuthConfirmPassword("");
              }}
              className="text-primary hover:underline hover:underline-offset-4"
            >
              {isLoginView ? "Don't have an account? Sign up" : "Already have an account? Log in"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full flex-col bg-slate-50">
      <header className="flex-shrink-0 border-b bg-white px-6 py-4">
        <div className="mx-auto flex w-full max-w-4xl items-center gap-3">
          <Avatar>
            <AvatarFallback className="bg-primary text-primary-foreground font-heading italic">
              <span className="italic">i</span>
            </AvatarFallback>
          </Avatar>
          <div>
            <h1 className="text-xl leading-none font-semibold tracking-tight italic">ira</h1>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleSaveChat} disabled={isSaving}>
              {isSaving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4" />}
              {isSaving ? "Saving..." : "Save"}
            </Button>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden bg-slate-50/50">
        <ScrollArea className="h-full w-full">
          <div className="mx-auto flex w-full max-w-4xl flex-col gap-4 p-6 pb-8">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex max-w-[80%] flex-col gap-0 ${
                  message.role === "user" ? "ml-auto items-end" : "mr-auto items-start"
                }`}
              >
                <div
                  className={`flex gap-3 ${
                    message.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"
                  }`}
                >
                  <Avatar className="h-8 w-8 shrink-0">
                    <AvatarFallback
                      className={
                        message.role === "user"
                          ? "bg-slate-200 text-slate-700"
                          : "bg-primary text-primary-foreground"
                      }
                    >
                      {message.role === "user" ? <User size={16} /> : <span className="font-heading italic">i</span>}
                    </AvatarFallback>
                  </Avatar>

                  <div
                    className={`rounded-2xl px-4 py-3 text-sm shadow-sm relative ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground rounded-tr-sm"
                        : "rounded-tl-sm border bg-white text-foreground"
                    }`}
                  >
                    {message.text}
                    {message.timestamp && (
                      <div className={`text-[10px] mt-1 opacity-50 ${message.role === "user" ? "text-right" : "text-left"}`}>
                        {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </div>
                    )}
                  </div>
                </div>
                {message.role === "assistant" && message.memoryTrace ? (
                  <div className="max-w-[min(100%,36rem)] pl-11">
                    <MemoryThisTurn trace={message.memoryTrace} />
                  </div>
                ) : null}
              </div>
            ))}
            {sending && (
              <div className="text-muted-foreground mr-auto flex items-center gap-2 text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                Thinking…
              </div>
            )}
            {error && (
              <p className="text-destructive mr-auto max-w-[80%] text-sm" role="alert">
                {error}
              </p>
            )}
          </div>
        </ScrollArea>
      </main>

      <footer className="flex-shrink-0 bg-transparent p-4 pb-8">
        <div className="mx-auto w-full max-w-3xl">
          <form onSubmit={handleSendMessage} className="relative flex items-center">
            <Input
              type="text"
              placeholder="Type your message..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              disabled={sending}
              className="h-12 flex-1 rounded-full border-slate-200 bg-slate-50 pr-14 pl-6 shadow-sm focus-visible:ring-1"
            />
            <Button
              type="submit"
              size="icon"
              disabled={!inputValue.trim() || sending}
              className="absolute right-1.5 h-9 w-9 rounded-full"
            >
              {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              <span className="sr-only">Send message</span>
            </Button>
          </form>
        </div>
      </footer>
    </div>
  );
}
