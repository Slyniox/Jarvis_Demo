const STORAGE_KEY = "jarvis.conversations.v1";

function now() {
  return Date.now();
}

export function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveConversations(conversations) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

export function createConversation() {
  const timestamp = now();
  return {
    id: crypto.randomUUID ? crypto.randomUUID() : `chat-${timestamp}-${Math.random()}`,
    title: "New conversation",
    messages: [],
    createdAt: timestamp,
    updatedAt: timestamp,
  };
}

export function upsertConversation(conversation) {
  const conversations = loadConversations();
  const index = conversations.findIndex(item => item.id === conversation.id);

  if (index >= 0) {
    conversations[index] = conversation;
  } else {
    conversations.push(conversation);
  }

  conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  saveConversations(conversations);
  return conversations;
}

export function deleteConversation(id) {
  const conversations = loadConversations().filter(item => item.id !== id);
  saveConversations(conversations);
  return conversations;
}

export function makeTitle(text) {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "New conversation";
  return clean.length > 42 ? `${clean.slice(0, 42).trimEnd()}…` : clean;
}
