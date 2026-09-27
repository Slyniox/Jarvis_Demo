import { streamChat } from "./api.js";
import {
  loadConversations,
  saveConversations,
  createConversation,
  upsertConversation,
  deleteConversation,
  makeTitle,
} from "./storage.js";
import { ChatUI } from "./ui.js";

const ui = new ChatUI();

let conversations = loadConversations();
let activeConversation = null;
let abortController = null;
let generationToken = null;

const SYSTEM_PROMPT_KEY = "jarvis-system-prompt";
const PIPELINE_KEY = "jarvis-pipeline";

function loadSystemPrompt() {
  return localStorage.getItem(SYSTEM_PROMPT_KEY) || "";
}

function saveSystemPrompt(value) {
  const prompt = value.trim();
  if (prompt) {
    localStorage.setItem(SYSTEM_PROMPT_KEY, prompt);
  } else {
    localStorage.removeItem(SYSTEM_PROMPT_KEY);
  }
}

function loadPipeline() {
  return localStorage.getItem(PIPELINE_KEY) === "fast" ? "fast" : "normal";
}

function savePipeline(value) {
  localStorage.setItem(PIPELINE_KEY, value === "fast" ? "fast" : "normal");
}

function messageId() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function ensureConversation() {
  if (activeConversation) return activeConversation;
  activeConversation = createConversation();
  conversations = upsertConversation(activeConversation);
  return activeConversation;
}

function selectConversation(id) {
  const found = conversations.find(item => item.id === id);
  if (!found) return;

  activeConversation = found;
  ui.setTitle(found.title);
  ui.clearMessages();

  if (!found.messages.length) {
    ui.showWelcome(true);
  } else {
    ui.showWelcome(false);
    for (const message of found.messages) {
      ui.addMessage(message.role, message.content, { messageId: message.id });
    }
  }

  ui.renderHistory(conversations, activeConversation.id);
  ui.closeSidebar();
  ui.focusInput();
}

function newConversation() {
  stopGeneration();

  activeConversation = createConversation();
  conversations = upsertConversation(activeConversation);

  ui.setTitle(activeConversation.title);
  ui.clearMessages();
  ui.showWelcome(true);
  ui.renderHistory(conversations, activeConversation.id);
  ui.closeSidebar();
  ui.focusInput();
}

function persistActive() {
  if (!activeConversation) return;
  activeConversation.updatedAt = Date.now();
  conversations = upsertConversation(activeConversation);
  ui.renderHistory(conversations, activeConversation.id);
}

function stopGeneration() {
  if (abortController) {
    abortController.abort();
  }
}

function deleteActiveConversation() {
  if (!activeConversation) return;

  stopGeneration();

  const id = activeConversation.id;
  conversations = deleteConversation(id);
  activeConversation = null;

  if (conversations.length) {
    selectConversation(conversations[0].id);
  } else {
    newConversation();
  }
}

function deleteMessage(id) {
  if (!activeConversation || !id) return;

  if (abortController) return;

  const index = activeConversation.messages.findIndex(message => message.id === id);
  if (index === -1) return;

  activeConversation.messages.splice(index, 1);
  persistActive();

  const row = ui.content.querySelector(`[data-message-id="${CSS.escape(id)}"]`);
  row?.remove();

  if (!activeConversation.messages.length) {
    ui.showWelcome(true);
  }
}

function beginEditMessage(id) {
  if (!activeConversation || abortController) return;

  const message = activeConversation.messages.find(item => item.id === id);
  if (!message || message.role !== "user") return;

  ui.input.value = message.content;
  ui.input.focus();
  ui.input.style.height = "auto";
  ui.input.style.height = `${Math.min(ui.input.scrollHeight, 160)}px`;
  ui.input.dataset.editingMessage = id;
  ui.input.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function editMessage(id, newText) {
  const index = activeConversation.messages.findIndex(item => item.id === id);
  if (index === -1) return;

  // Editing a user turn discards that turn and everything after it.
  activeConversation.messages.splice(index);
  persistActive();

  // Re-render the truncated conversation before regenerating.
  ui.clearMessages();
  if (activeConversation.messages.length) {
    for (const message of activeConversation.messages) {
      ui.addMessage(message.role, message.content, { messageId: message.id });
    }
  } else {
    ui.showWelcome(true);
  }

  ui.input.dataset.editingMessage = "";
  await sendMessage(newText);
}

async function regenerateMessage(id) {
  if (!activeConversation || abortController) return;

  const index = activeConversation.messages.findIndex(item => item.id === id);
  if (index === -1) return;

  const message = activeConversation.messages[index];
  if (message.role !== "assistant") return;

  // The assistant response and all later turns are replaced.
  const userIndex = index - 1;
  if (userIndex < 0 || activeConversation.messages[userIndex].role !== "user") return;

  activeConversation.messages.splice(index);
  persistActive();

  ui.clearMessages();
  for (const item of activeConversation.messages) {
    ui.addMessage(item.role, item.content, { messageId: item.id });
  }

  await generateAssistant();
}

async function generateAssistant() {
  if (!activeConversation || abortController) return;

  const generationConversation = activeConversation;
  const token = {};
  generationToken = token;

  // Create the activity first so it appears above the response.
  const activity = ui.createActivity();

  const assistantId = messageId();
  const assistant = ui.addMessage("assistant", "", {
    messageId: assistantId,
    streaming: true,
  });

  ui.insertActivityBefore(activity, assistant.row);
  ui.beginAutoFollow();

  abortController = new AbortController();
  ui.setBusy(true);

  try {
    const systemPrompt = loadSystemPrompt();

    const messagesForModel = generationConversation.messages.map(({ role, content }) => ({
      role,
      content,
    }));

    if (systemPrompt) {
      messagesForModel.unshift({
        role: "system",
        content: systemPrompt,
      });
    }

    const answer = await streamChat(
      messagesForModel,
      (_delta, fullText) => {
        if (generationToken === token) assistant.setText(fullText);
      },
      (status) => {
        if (generationToken === token) activity.update(status);
      },
      abortController.signal,
      loadPipeline(),
    );

    if (generationToken === token && activeConversation === generationConversation) {
      assistant.finish(answer);
      activity.finish();

      if (answer) {
        generationConversation.messages.push({
          id: assistantId,
          role: "assistant",
          content: answer,
        });
        persistActive();
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      if (generationToken === token) {
        assistant.finish(error.partialText ?? assistant.getText());
        activity.finish();
      }
    } else if (generationToken === token) {
      activity.fail(error.message);
      assistant.error(`Unable to reach Jarvis: ${error.message}`);
    }
  } finally {
    if (generationToken === token) {
      abortController = null;
      generationToken = null;
      ui.setBusy(false);
      ui.focusInput();
    }
  }
}

async function sendMessage(text) {
  if (!text || abortController) return;

  ensureConversation();

  if (!activeConversation.messages.some(message => message.role === "user")) {
    activeConversation.title = makeTitle(text);
    ui.setTitle(activeConversation.title);
  }

  activeConversation.messages.push({
    id: messageId(),
    role: "user",
    content: text,
  });

  persistActive();
  ui.addMessage("user", text, {
    messageId: activeConversation.messages.at(-1).id,
  });

  await generateAssistant();
}

function renameActiveConversation() {
  if (!activeConversation || abortController) return;

  const proposed = window.prompt("Conversation name:", activeConversation.title || "");
  if (proposed === null) return;

  const name = proposed.trim();
  if (!name) return;

  activeConversation.title = name.slice(0, 100);
  activeConversation.updatedAt = Date.now();
  conversations = upsertConversation(activeConversation);
  ui.setTitle(activeConversation.title);
  ui.renderHistory(conversations, activeConversation.id);
}

function wireEvents() {
  const effortSelector = document.getElementById("effort-selector");
  const systemPromptButton = document.getElementById("system-prompt-button");
  const systemPromptPanel = document.getElementById("system-prompt-panel");
  const systemPromptInput = document.getElementById("system-prompt-input");
  const systemPromptSave = document.getElementById("system-prompt-save");
  const systemPromptClear = document.getElementById("system-prompt-clear");

  if (effortSelector) {
    effortSelector.value = loadPipeline();
    effortSelector.addEventListener("change", () => {
      savePipeline(effortSelector.value);
    });
  }

  if (systemPromptInput) {
    systemPromptInput.value = loadSystemPrompt();
  }

  systemPromptButton?.addEventListener("click", () => {
    const visible = systemPromptPanel?.classList.toggle("visible");
    if (visible) {
      systemPromptInput?.focus();
      systemPromptInput?.setSelectionRange(
        systemPromptInput.value.length,
        systemPromptInput.value.length,
      );
    }
  });

  systemPromptSave?.addEventListener("click", () => {
    saveSystemPrompt(systemPromptInput?.value || "");
    systemPromptPanel?.classList.remove("visible");
  });

  systemPromptClear?.addEventListener("click", () => {
    if (systemPromptInput) systemPromptInput.value = "";
    saveSystemPrompt("");
    systemPromptPanel?.classList.remove("visible");
  });

  document.getElementById("composer").addEventListener("submit", async event => {
    event.preventDefault();

    if (abortController) {
      stopGeneration();
      return;
    }

    const text = ui.input.value.trim();
    if (!text) return;

    const editingId = ui.input.dataset.editingMessage;
    ui.input.value = "";
    ui.input.style.height = "auto";
    ui.input.dataset.editingMessage = "";

    if (editingId) {
      await editMessage(editingId, text);
    } else {
      await sendMessage(text);
    }
  });

  ui.input.addEventListener("input", () => {
    ui.input.style.height = "auto";
    ui.input.style.height = `${Math.min(ui.input.scrollHeight, 160)}px`;
  });

  ui.input.addEventListener("keydown", event => {
    if (event.key === "Escape" && ui.input.dataset.editingMessage) {
      ui.input.value = "";
      ui.input.dataset.editingMessage = "";
      ui.input.style.height = "auto";
      return;
    }

    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      document.getElementById("composer").requestSubmit();
    }
  });

  document.getElementById("new-chat").addEventListener("click", newConversation);


  // Rename by clicking the conversation title in the top bar.
  ui.title.addEventListener("click", renameActiveConversation);
  ui.title.title = "Rename conversation";

  document.getElementById("open-sidebar").addEventListener("click", () => ui.openSidebar());
  document.getElementById("sidebar-backdrop").addEventListener("click", () => ui.closeSidebar());

  ui.history.addEventListener("click", event => {
    const deleteButton = event.target.closest("[data-delete-conversation]");
    if (deleteButton) {
      event.stopPropagation();
      const id = deleteButton.dataset.deleteConversation;

      if (id === activeConversation?.id) {
        deleteActiveConversation();
      } else {
        conversations = deleteConversation(id);
        ui.renderHistory(conversations, activeConversation?.id);
      }
      return;
    }

    const item = event.target.closest(".conversation-item");
    if (item) selectConversation(item.dataset.id);
  });

  ui.content.addEventListener("click", event => {
    const deleteButton = event.target.closest("[data-delete-message]");
    if (deleteButton) {
      deleteMessage(deleteButton.dataset.deleteMessage);
      return;
    }

    const editButton = event.target.closest("[data-edit-message]");
    if (editButton) {
      beginEditMessage(editButton.dataset.editMessage);
      return;
    }

    const regenerateButton = event.target.closest("[data-regenerate-message]");
    if (regenerateButton) {
      regenerateMessage(regenerateButton.dataset.regenerateMessage);
      return;
    }

    const prompt = event.target.closest("[data-prompt]");
    if (prompt) {
      ui.input.value = prompt.dataset.prompt;
      ui.input.style.height = "auto";
      ui.input.style.height = `${Math.min(ui.input.scrollHeight, 160)}px`;
      ui.focusInput();
    }
  });
}

function boot() {
  wireEvents();

  if (conversations.length) {
    conversations.sort((a, b) => b.updatedAt - a.updatedAt);
    saveConversations(conversations);
    selectConversation(conversations[0].id);
  } else {
    newConversation();
  }
}

boot();


function getVoiceUrl() {
  if (location.protocol === "https:") {
    return `wss://${location.host}/voice`;
  }
  return `ws://${location.hostname}:9100`;
}
const VOICE_URL = getVoiceUrl();
let voiceSocket=null, voiceContext=null, voiceSource=null, voiceProcessor=null, voiceStream=null, voiceActive=false;
const VOICE_OUTPUT_SAMPLE_RATE=24000;
let voicePlaybackContext=null, voicePlaybackSampleRate=VOICE_OUTPUT_SAMPLE_RATE, voicePlaybackNextTime=0, voicePlaybackPendingDone=false;
const voicePlaybackSources=new Set();
// Small overlap + crossfade removes clicks/pops at PCM chunk boundaries.
// 5 ms is short enough to preserve speech transients while smoothing joins.
const VOICE_CHUNK_CROSSFADE=0.005;

function voiceStatus(text){document.getElementById("voice-mode-status")?.replaceChildren(document.createTextNode(text));}
function setVoiceActive(active){
  voiceActive=active;
  document.getElementById("voice-hold-button")?.classList.toggle("active",active);
  document.getElementById("voice-mode-page")?.classList.toggle("listening",active);
  voiceStatus(active?"Listening…":"Ready");
  const i=document.getElementById("voice-mode-instruction");
  if(i)i.textContent=active?"Release to send":"Hold to speak";
}
function f32pcm(x){const b=new Int16Array(x.length);for(let i=0;i<x.length;i++){const s=Math.max(-1,Math.min(1,x[i]));b[i]=s<0?s*32768:s*32767}return b;}
function ensureVoicePlaybackContext(){
  if(!voicePlaybackContext)voicePlaybackContext=new AudioContext();
  return voicePlaybackContext.resume().then(()=>voicePlaybackContext);
}
function stopVoicePlayback(){
  for(const source of voicePlaybackSources){try{source.stop()}catch{}}
  voicePlaybackSources.clear();
  voicePlaybackNextTime=0;
  voicePlaybackPendingDone=false;
}
function handleVoiceAudioStart(m){
  const sampleRate=Number(m.sample_rate);
  // Supertonic outputs PCM16 at 24 kHz. Respect the announced rate when present,
  // but keep 24 kHz as the known-good fallback for this pipeline.
  voicePlaybackSampleRate=Number.isFinite(sampleRate)&&sampleRate>0?sampleRate:VOICE_OUTPUT_SAMPLE_RATE;
  stopVoicePlayback();
  if(!voicePlaybackContext)voicePlaybackContext=new AudioContext();
  voicePlaybackContext.resume().catch(()=>{});
  voicePlaybackPendingDone=false;
  voiceStatus("Speaking…");
}
function handleVoiceAudioChunk(data){
  if(!(data instanceof ArrayBuffer)||data.byteLength<2)return;
  const ctx=voicePlaybackContext;
  if(!ctx)return;
  const pcm=new Int16Array(data);
  const samples=new Float32Array(pcm.length);
  for(let i=0;i<pcm.length;i++)samples[i]=pcm[i]/32768;
  const buffer=ctx.createBuffer(1,samples.length,voicePlaybackSampleRate);
  buffer.copyToChannel(samples,0);

  const source=ctx.createBufferSource();
  source.buffer=buffer;

  // Crossfade the beginning/end of each chunk instead of hard-cutting from
  // one independently scheduled AudioBufferSourceNode to the next.
  const gain=ctx.createGain();
  source.connect(gain);
  gain.connect(ctx.destination);

  const now=ctx.currentTime;
  const minLead=0.010;
  const overlap=Math.min(VOICE_CHUNK_CROSSFADE, buffer.duration * 0.25);
  const startTime=Math.max(voicePlaybackNextTime-overlap, now+minLead);
  const endTime=startTime+buffer.duration;

  gain.gain.setValueAtTime(0, startTime);
  gain.gain.linearRampToValueAtTime(1, startTime+overlap);
  gain.gain.setValueAtTime(1, Math.max(startTime+overlap, endTime-overlap));
  gain.gain.linearRampToValueAtTime(0, endTime);

  voicePlaybackNextTime=endTime-overlap;
  voicePlaybackSources.add(source);
  source.onended=()=>{
    voicePlaybackSources.delete(source);
    if(voicePlaybackPendingDone&&voicePlaybackSources.size===0){
      voicePlaybackPendingDone=false;
      voicePlaybackNextTime=0;
      voiceStatus("Ready");
    }
  };
  source.start(startTime);
}
function handleVoiceAudioDone(){
  if(voicePlaybackSources.size===0){
    voicePlaybackNextTime=0;
    voiceStatus("Ready");
  }else{
    voicePlaybackPendingDone=true;
    voiceStatus("Speaking…");
  }
}
async function connectVoice(){
  if(voiceSocket?.readyState===WebSocket.OPEN)return;
  voiceStatus("Connecting…");
  await new Promise((resolve,reject)=>{
    const s=new WebSocket(VOICE_URL);s.binaryType="arraybuffer";
    const t=setTimeout(()=>{s.close();reject(new Error("Voice proxy timeout"));},8000);
    s.onopen=()=>{clearTimeout(t);voiceSocket=s;resolve()};
    s.onerror=()=>{clearTimeout(t);reject(new Error("Could not connect to voice proxy on port 9100"))};
    s.onclose=()=>{if(voiceSocket===s)voiceSocket=null;if(voiceActive)setVoiceActive(false)};
    s.onmessage=e=>{
      if(typeof e.data!=="string"){handleVoiceAudioChunk(e.data);return;}
      try{
        const m=JSON.parse(e.data);
        if(m.type==="status")voiceStatus(m.text||m.state||"Working…");
        if(m.type==="result"){
          const el=document.getElementById("voice-mode-transcript");
          if(el)el.textContent=m.text||"";
          if(m.final===true)voiceStatus("Thinking…");
        }else if(m.type==="transcript"){
          const el=document.getElementById("voice-mode-transcript");
          if(el)el.textContent=m.text||"";
        }else if(m.type==="audio_start"){
          handleVoiceAudioStart(m);
        }else if(m.type==="audio_done"){
          handleVoiceAudioDone();
        }
      }catch(err){console.warn("[Voice] Invalid message",err);}
    };
  });
}
async function startVoice(){
  if(voiceActive)return;
  try{
    await connectVoice();
    voiceStream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
    voiceContext=new AudioContext();
    await voiceContext.resume();
    await ensureVoicePlaybackContext();
    voiceSource=voiceContext.createMediaStreamSource(voiceStream);
    voiceProcessor=voiceContext.createScriptProcessor(4096,1,1);
    voiceProcessor.onaudioprocess=e=>{
      if(voiceActive&&voiceSocket?.readyState===WebSocket.OPEN)voiceSocket.send(f32pcm(e.inputBuffer.getChannelData(0)).buffer);
    };
    voiceSource.connect(voiceProcessor);voiceProcessor.connect(voiceContext.destination);setVoiceActive(true);
  }catch(e){console.error("[Voice]",e);voiceStatus(e.message||"Microphone unavailable");}
}
function stopVoice(){
  if(!voiceActive&&!voiceStream)return;setVoiceActive(false);
  try{voiceProcessor?.disconnect()}catch{} try{voiceSource?.disconnect()}catch{}
  voiceProcessor=null;voiceSource=null;
  if(voiceContext){voiceContext.close().catch(()=>{});voiceContext=null}
  if(voiceStream){voiceStream.getTracks().forEach(t=>t.stop());voiceStream=null}
  if(voiceSocket?.readyState===WebSocket.OPEN)try{voiceSocket.send(JSON.stringify({type:"end"}))}catch{}
}
function openVoice(){document.getElementById("voice-mode-page")?.classList.add("visible");document.body.classList.add("voice-mode-open");}
function closeVoice(){stopVoice();document.getElementById("voice-mode-page")?.classList.remove("visible");document.body.classList.remove("voice-mode-open");}
function wireVoice(){
  document.getElementById("voice-mode-button")?.addEventListener("click",openVoice);
  document.getElementById("voice-back-button")?.addEventListener("click",closeVoice);
  const b=document.getElementById("voice-hold-button");if(!b)return;
  const down=e=>{e.preventDefault();b.setPointerCapture?.(e.pointerId);startVoice()};
  const up=e=>{e.preventDefault();stopVoice()};
  b.addEventListener("pointerdown",down);b.addEventListener("pointerup",up);b.addEventListener("pointercancel",up);
}
wireVoice();
