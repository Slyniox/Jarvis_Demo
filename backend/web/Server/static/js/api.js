export async function streamChat(messages, onDelta, onStatus, signal, pipeline = "normal") {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
    body: JSON.stringify({ messages, stream: true, pipeline }),
    signal,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); if (body?.error) detail = body.error; } catch {}
    throw new Error(detail);
  }
  if (!response.body) throw new Error("The server did not return a streaming response.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";

  const processLine = (rawLine) => {
    const line = rawLine.trim();
    if (!line || !line.startsWith("data:")) return false;
    const data = line.slice(5).trim();
    if (data === "[DONE]") return true;

    try {
      const chunk = JSON.parse(data);
      if (chunk.type === "status") {
        if (typeof onStatus === "function") onStatus(chunk);
        return false;
      }
      if (chunk.error) throw new Error(chunk.error.message || "Streaming error");
      const delta = chunk.choices?.[0]?.delta?.content;
      if (typeof delta === "string" && delta.length) {
        fullText += delta;
        onDelta(delta, fullText);
      }
    } catch (error) {
      if (error instanceof SyntaxError) return false;
      throw error;
    }
    return false;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) if (processLine(line)) {
        await reader.cancel();
        return fullText;
      }
    }
    if (buffer) processLine(buffer);
    return fullText;
  } catch (error) {
    if (error?.name === "AbortError") error.partialText = fullText;
    throw error;
  } finally {
    reader.releaseLock();
  }
}
