import { renderMarkdown } from "./markdown.js";

export class ChatUI {
  constructor() {
    this.content = document.getElementById("chat-content");
    this.scroll = document.getElementById("chat-scroll");
    this.title = document.getElementById("conversation-title");
    this.history = document.getElementById("conversation-list");
    this.input = document.getElementById("message-input");
    this.sendButton = document.getElementById("send-button");
    this.sidebar = document.getElementById("sidebar");
    this.backdrop = document.getElementById("sidebar-backdrop");

    // Follow live output until the user scrolls away. Once detached,
    // streaming/status updates never pull the viewport back down.
    this.autoFollow = true;
    this.jumpButton = document.getElementById("jump-to-latest");

    this.scroll.addEventListener("scroll", () => {
      if (this.isNearBottom(140)) {
        this.autoFollow = true;
        this.hideJumpButton();
      } else {
        this.autoFollow = false;
        this.showJumpButton();
      }
    }, { passive: true });

    this.jumpButton?.addEventListener("click", () => {
      this.autoFollow = true;
      this.scrollToBottom(true);
      this.hideJumpButton();
    });
  }

  setTitle(title) {
    this.title.textContent = title;
  }

  showWelcome(show) {
    if (show) {
      this.content.innerHTML = `
        <section class="welcome" id="welcome">
          <div class="orb"><div class="orb-core">J</div></div>
          <h1>Good to see you.</h1>
          <p>Ask Jarvis anything. Your conversation stays on your local system.</p>
          <div class="suggestions">
            <button type="button" data-prompt="Search my diary">Search my diary</button>
            <button type="button" data-prompt="Search the web">Search the web</button>
          </div>
        </section>
      `;
    }
  }

  clearMessages() {
    this.content.innerHTML = "";
  }

  createActivity() {
    const block = document.createElement("details");
    block.className = "activity-block";
    block.open = true;

    const summary = document.createElement("summary");
    summary.className = "activity-summary";

    const icon = document.createElement("span");
    icon.className = "activity-icon";
    icon.textContent = "✦";

    const summaryText = document.createElement("span");
    summaryText.className = "activity-summary-text";
    summaryText.textContent = "Working on your request";

    const chevron = document.createElement("span");
    chevron.className = "activity-chevron";
    chevron.textContent = "⌄";

    summary.append(icon, summaryText, chevron);

    const timeline = document.createElement("div");
    timeline.className = "activity-timeline";

    block.append(summary, timeline);
    this.content.appendChild(block);
    this.scrollToBottom();

    const labels = {
      planning: "Planning",
      diary: "Searching your diary",
      web: "Searching the web",
      reranking: "Reviewing results",
      thinking: "Thinking",
      generating: "Generating response",
    };

    const entries = new Map();
    const startedAt = performance.now();

    const formatDuration = (ms) => {
      const seconds = ms / 1000;
      if (seconds < 1) return "<1s";
      return `${seconds.toFixed(1)}s`;
    };

    const updateSummary = () => {
      const active = [...entries.values()].find(entry => entry.state === "started");
      summaryText.textContent = active
        ? active.label
        : `Worked for ${formatDuration(performance.now() - startedAt)}`;
    };

    const renderEntry = (event) => {
      if (!event?.stage) return;

      let entry = entries.get(event.stage);
      if (!entry) {
        const row = document.createElement("div");
        row.className = "activity-entry";

        const marker = document.createElement("span");
        marker.className = "activity-entry-marker";

        const body = document.createElement("div");
        body.className = "activity-entry-body";

        const label = document.createElement("div");
        label.className = "activity-entry-label";

        const detail = document.createElement("div");
        detail.className = "activity-entry-detail";

        body.append(label, detail);
        row.append(marker, body);
        timeline.appendChild(row);

        entry = { row, marker, label, detail, labelText: labels[event.stage] || event.stage, state: event.state };
        entries.set(event.stage, entry);
      }

      entry.state = event.state;
      entry.label.textContent = entry.labelText;
      entry.row.classList.toggle("started", event.state === "started");
      entry.row.classList.toggle("completed", event.state === "completed");
      entry.row.classList.toggle("failed", event.state === "failed");
      entry.marker.textContent = event.state === "completed" ? "✓" : event.state === "failed" ? "!" : "●";

      if (event.detail) {
        entry.detail.textContent = event.detail;
        entry.detail.hidden = false;
      } else {
        entry.detail.textContent = "";
        entry.detail.hidden = true;
      }

      updateSummary();
      this.scrollToBottom();
    };

    return {
      update: renderEntry,
      finish: () => {
        for (const entry of entries.values()) {
          if (entry.state === "started") {
            entry.state = "completed";
            entry.marker.textContent = "✓";
            entry.row.classList.remove("started");
            entry.row.classList.add("completed");
          }
        }
        updateSummary();
        block.open = false;
        this.scrollToBottom();
      },
      fail: (detail) => {
        if (detail) {
          const row = document.createElement("div");
          row.className = "activity-entry failed";
          const marker = document.createElement("span");
          marker.className = "activity-entry-marker";
          marker.textContent = "!";
          const body = document.createElement("div");
          body.className = "activity-entry-body";
          const label = document.createElement("div");
          label.className = "activity-entry-label";
          label.textContent = "Request failed";
          const detailNode = document.createElement("div");
          detailNode.className = "activity-entry-detail";
          detailNode.textContent = detail;
          body.append(label, detailNode);
          row.append(marker, body);
          timeline.appendChild(row);
        }
        summaryText.textContent = "Something went wrong";
        block.open = true;
        this.scrollToBottom();
      },
      getElement: () => block,
    };
  }

  insertActivityBefore(activity, element) {
    if (!activity || !element) return;
    this.content.insertBefore(activity.getElement(), element);
    this.scrollToBottom();
  }

  addMessage(role, text = "", options = {}) {
    this.showWelcome(false);

    const row = document.createElement("div");
    row.className = `message-row ${role}`;
    if (options.messageId) row.dataset.messageId = options.messageId;

    const message = document.createElement("article");
    message.className = `message ${role}`;

    const header = document.createElement("div");
    header.className = "message-header";

    const roleLabel = document.createElement("div");
    roleLabel.className = "message-role";
    roleLabel.textContent = role === "user" ? "YOU" : "JARVIS";

    const actions = document.createElement("div");
    actions.className = "message-actions";

    if (role === "user") {
      const editButton = document.createElement("button");
      editButton.className = "message-action";
      editButton.type = "button";
      editButton.dataset.editMessage = options.messageId || "";
      editButton.setAttribute("aria-label", "Edit message");
      editButton.title = "Edit message";
      editButton.textContent = "✎";
      actions.appendChild(editButton);
    }

    if (role === "assistant") {
      const regenerateButton = document.createElement("button");
      regenerateButton.className = "message-action";
      regenerateButton.type = "button";
      regenerateButton.dataset.regenerateMessage = options.messageId || "";
      regenerateButton.setAttribute("aria-label", "Regenerate response");
      regenerateButton.title = "Regenerate response";
      regenerateButton.textContent = "↻";
      actions.appendChild(regenerateButton);
    }

    const deleteButton = document.createElement("button");
    deleteButton.className = "message-action";
    deleteButton.type = "button";
    deleteButton.dataset.deleteMessage = options.messageId || "";
    deleteButton.setAttribute("aria-label", "Delete message");
    deleteButton.title = "Delete message";
    deleteButton.textContent = "×";

    actions.appendChild(deleteButton);
    header.append(roleLabel, actions);

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (options.streaming) {
      bubble.classList.add("streaming");
      bubble.textContent = text;
    } else {
      bubble.innerHTML = renderMarkdown(text);
    }

    message.append(header, bubble);
    row.appendChild(message);
    this.content.appendChild(row);
    this.scrollToBottom(true);

    return {
      row,
      bubble,
      setText: (value) => {
        if (options.streaming) {
          // Streaming deliberately stays cheap: render only when the browser
          // paints, while preserving Markdown-safe output.
          bubble.innerHTML = renderMarkdown(value);
          bubble.classList.add("streaming");
        } else {
          bubble.innerHTML = renderMarkdown(value);
        }
        this.scrollToBottom();
      },
      finish: (finalText = null) => {
        bubble.classList.remove("streaming");
        if (finalText !== null) {
          bubble.innerHTML = renderMarkdown(finalText);
        }
      },
      getText: () => bubble.textContent || "",
      error: (value) => {
        bubble.textContent = value;
        bubble.classList.add("message-error");
        bubble.classList.remove("streaming");
      },
    };
  }

  renderHistory(conversations, activeId) {
    this.history.innerHTML = "";

    for (const conversation of conversations) {
      const wrapper = document.createElement("div");
      wrapper.className = `conversation-item-wrapper ${conversation.id === activeId ? "active" : ""}`;

      const button = document.createElement("button");
      button.className = "conversation-item";
      button.type = "button";
      button.dataset.id = conversation.id;

      const icon = document.createElement("span");
      icon.className = "conversation-icon";
      icon.textContent = "◆";

      const name = document.createElement("span");
      name.className = "conversation-name";
      name.textContent = conversation.title || "New conversation";

      const deleteButton = document.createElement("button");
      deleteButton.className = "conversation-delete";
      deleteButton.type = "button";
      deleteButton.dataset.deleteConversation = conversation.id;
      deleteButton.setAttribute("aria-label", `Delete ${conversation.title || "conversation"}`);
      deleteButton.title = "Delete conversation";
      deleteButton.textContent = "×";

      button.append(icon, name);
      wrapper.append(button, deleteButton);
      this.history.appendChild(wrapper);
    }
  }

  setBusy(busy) {
    // The input remains disabled during generation, while the button remains
    // clickable so it can act as the Stop control.
    this.input.disabled = busy;
    this.sendButton.disabled = false;
    this.sendButton.classList.toggle("generating", busy);
    this.sendButton.setAttribute("aria-label", busy ? "Stop generation" : "Send message");
    this.sendButton.title = busy ? "Stop generation" : "Send message";
    this.input.placeholder = busy ? "Jarvis is thinking…" : "Message Jarvis…";
  }

  focusInput() {
    if (!this.input.disabled) this.input.focus();
  }

  isNearBottom(threshold = 140) {
    return this.scroll.scrollHeight - this.scroll.scrollTop - this.scroll.clientHeight <= threshold;
  }

  scrollToBottom(force = false) {
    if (!force && !this.autoFollow) return;

    requestAnimationFrame(() => {
      if (!force && !this.autoFollow) return;
      this.scroll.scrollTop = this.scroll.scrollHeight;
      this.hideJumpButton();
    });
  }

  beginAutoFollow() {
    this.autoFollow = true;
    this.hideJumpButton();
    this.scrollToBottom(true);
  }

  showJumpButton() {
    this.jumpButton?.classList.add("visible");
  }

  hideJumpButton() {
    this.jumpButton?.classList.remove("visible");
  }

  openSidebar() {
    this.sidebar.classList.add("open");
    this.backdrop.classList.add("visible");
  }

  closeSidebar() {
    this.sidebar.classList.remove("open");
    this.backdrop.classList.remove("visible");
  }
}
