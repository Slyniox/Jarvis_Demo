#!/usr/bin/env python3

import asyncio
import json

import httpx

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from contextlib import asynccontextmanager

from launcher import load_config, manager
from create_context import create_context
from request_context import prepare_model_request



FAST_PLANNER_PROMPT = """Tu es le planificateur RAPIDE de Jarvis.

Ta mission est de décider si la question de l'utilisateur nécessite :
- une recherche dans son journal ;
- une recherche sur le web ;
- du raisonnement supplémentaire du modèle.

PRIORITÉ ABSOLUE : MINIMISER LA LATENCE.

Réponds directement dès que possible. N'effectue aucune recherche si elle n'est pas réellement nécessaire. En cas de doute, ne recherche pas.

RAISONNEMENT
need_reasoning=false par défaut.
Passe à true uniquement si l'utilisateur demande explicitement une analyse, un raisonnement approfondi, ou si la tâche nécessite clairement un raisonnement important.

JOURNAL
Utilise le journal uniquement lorsque la réponse dépend d'informations personnelles de l'utilisateur : expériences, activités, personnes, projets, événements, souvenirs, etc.
Si nécessaire, génère UNE SEULE requête de recherche en français.
Ne génère jamais plusieurs variantes.

WEB
Utilise le web uniquement lorsque des informations externes ou actuelles sont réellement nécessaires.
Si nécessaire, génère UNE SEULE requête concise.
Ne génère jamais plusieurs variantes.

Les deux recherches peuvent être activées uniquement si elles sont toutes les deux réellement nécessaires.

Pour les questions larges concernant le journal, génère une seule requête couvrant le sujet principal plutôt que plusieurs requêtes spécialisées.

Retourne UNIQUEMENT ce JSON valide :

{
  "need_diary": boolean,
  "need_web": boolean,
  "need_reasoning": boolean,
  "diary_queries": string[],
  "web_queries": string[]
}

Contraintes :
- maximum 1 diary query ;
- maximum 1 web query ;
- si need_diary=false, diary_queries=[] ;
- si need_web=false, web_queries=[] ;
- toutes les diary queries sont en français ;
- aucun texte en dehors du JSON."""



PLANNER_PROMPT = """
You are Jarvis's retrieval planner.

Your ONLY tasks are to decide whether reasoning is required and wether Jarvis needs to search:
- the user's diary
- the web

and to generate retrieval queries.

Retrieval is expensive. If you are uncertain whether a retrieval source is needed, do not use it.

Return ONLY a valid JSON object.

Schema:

{
  "need_reasoning": boolean,
  "need_diary": boolean,
  "need_web": boolean,
  "diary_queries": string[],
  "web_queries": string[]
}

The response must contain nothing except the JSON.

--------------------------------------------------
Decision process
--------------------------------------------------

Step 1. Decide whether reasoning is required.

Set need_reasoning = true ONLY when the final answer would significantly
benefit from multi-step reasoning, analysis, deduction, planning,
comparison, or resolving ambiguity.

Set need_reasoning = false when the answer can be produced accurately and
directly without substantial internal reasoning.

Use need_reasoning = true for:
- complex programming or architecture decisions
- multi-step mathematical or logical problems
- comparing multiple options and making a justified recommendation
- analyzing conflicting or incomplete information
- planning complex tasks
- problems requiring several pieces of information to be combined
- questions explicitly asking for detailed reasoning, analysis, or justification

Use need_reasoning = false for:
- simple factual questions
- straightforward programming questions
- simple calculations
- translations
- grammar or spelling corrections
- summarization
- rewriting or formatting
- simple retrieval-based questions where the answer is directly contained
  in the retrieved information
- casual conversation

The goal is NOT to determine whether the question is difficult in general.
The goal is to determine whether additional reasoning by the final model
WOULD materially improve the quality of the answer.

When uncertain, prefer need_reasoning = false.

--------------------------------------------------

Step 2. Decide whether the diary is required.

Set need_diary = true ONLY if the answer depends on:
- previous conversations
- the user's memories
- any personal information about the user
- previous decisions
- previous projects
- things the user has done
- any person the user explicitely mentions
- the prompt explicitely asking for diary retrieval

Otherwise set need_diary = false.

General knowledge MUST NOT use the diary.

--------------------------------------------------

Step 3. Decide whether the web is required.

Set need_web = true ONLY if the answer depends on:
- current information
- recent events
- news
- weather
- live data
- information unavailable from the model itself
- the prompt explicitely asking for web retrieval

Do NOT use the web for:
- programming questions
- mathematics
- science
- language

If the model already knows the answer, do NOT use the web.

--------------------------------------------------

Step 4. Generate retrieval queries.

If need_diary is false:
diary_queries MUST be [].

If need_web is false:
web_queries MUST be [].

If a retrieval source is used:
generate 1 precise query.

Generate a second query only if it retrieves different information.

Never generate more than 3 queries.

--------------------------------------------------

Search query rules
--------------------------------------------------

Every query must:
- be self-contained
- preserve all important context
- contain all important names
- never contain pronouns
- never depend on another query
- be suitable for semantic search

Bad:
"activities"

Good:
"Activities during Sicily trip"

Bad:
"Eagle"

Good:
"Decision about speculative decoding using Eagle"

Bad:
"discussion"

Good:
"Discussion about Qwen embeddings"

Prefer one detailed query over several short ones.

--------------------------------------------------

Consistency rules
--------------------------------------------------

If need_diary is false:
diary_queries MUST be [].

If need_web is false:
web_queries MUST be [].

If diary_queries is not empty:
need_diary MUST be true.

If web_queries is not empty:
need_web MUST be true.

--------------------------------------------------

Examples
--------------------------------------------------

User:
Who is the president of the US?

Output:
{
  "need_reasoning": false,
  "need_diary": false,
  "need_web": true,
  "diary_queries": [],
  "web_queries": [US president]
}

User:
What's the weather tomorrow?

Output:
{
  "need_reasoning": false,
  "need_diary": false,
  "need_web": true,
  "diary_queries": [],
  "web_queries": [
    "Weather forecast tomorrow"
  ]
}

User:
How do I write a binary search in C?

Output:
{
  "need_reasoning": true,
  "need_diary": false,
  "need_web": false,
  "diary_queries": [],
  "web_queries": []
}

User:
Quelles activités est-ce que j'ai fait pendant mon voyage en Sicile?

Output:
{
  "need_reasoning": false,
  "need_diary": true,
  "need_web": false,
  "diary_queries": [
    "Voyage en Sicile",
    "Activités voyage en Sicile",
  ],
  "web_queries": []
}

User:
Qu'est-ce qu'on a décidé à propos du décodage spéculatif?

Output:
{
  "need_reasoning": true,
  "need_diary": true,
  "need_web": false,
  "diary_queries": [
    "Décision décodage spéculatif"
  ],
  "web_queries": []
}
"""


################################################################################
# Configuration
################################################################################

config = load_config()
LLAMA_URL = (
    f"http://{config['llama']['client_host']}:{config['llama']['port']}"
)



from dataclasses import dataclass


@dataclass
class PipelineConfig:
    planner_prompt: str
    top_k: int
    max_full_entries: int
    max_summaries: int
    max_diary_queries: int
    max_web_queries: int


NORMAL_PIPELINE = PipelineConfig(
    planner_prompt=PLANNER_PROMPT,
    **config["pipelines"]["normal"],
)


FAST_PIPELINE = PipelineConfig(
    planner_prompt=FAST_PLANNER_PROMPT,
    **config["pipelines"]["fast"],
)


PIPELINES = {
    "normal": NORMAL_PIPELINE,
    "fast": FAST_PIPELINE,
}


################################################################################
# Status events
################################################################################

def send_status(send_event, stage, state="started", detail=None):
    """
    Send a status event to the frontend.

    Parameters
    ----------
    send_event : callable
        Function used to send the event.

    stage : str
        Current stage, e.g. "planning", "diary", "web", "thinking".

    state : str
        Event state: "started", "completed", or "failed".

    detail : str, optional
        Optional human-readable detail.
    """

    event = {
        "type": "status",
        "stage": stage,
        "state": state,
    }

    if detail is not None:
        event["detail"] = detail

    send_event(event)


################################################################################
# Stopping
################################################################################

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("[Proxy] Starting...")

    await asyncio.to_thread(manager.ensure)

    yield

    print("[Proxy] Shutting down...")

    manager.stop()


app = FastAPI(lifespan=lifespan)


################################################################################
# Normal completion
################################################################################

async def normal_completion(body):

    async with httpx.AsyncClient(timeout=None) as client:

        response = await client.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json=body,
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get(
                "content-type",
                "application/json",
            ),
        )


################################################################################
# Streaming completion
################################################################################

async def stream_completion(body, pipeline):

    ###########################################################################
    # Request-local event queue
    ###########################################################################

    queue = asyncio.Queue()

    loop = asyncio.get_running_loop()

    ###########################################################################
    # Function exposed to create_context()
    ###########################################################################

    def send_event(event):

        loop.call_soon_threadsafe(
            queue.put_nowait,
            event,
        )

    ###########################################################################
    # Build context in background
    ###########################################################################

    async def build_context():

        try:

            memory_context, need_reasoning = await asyncio.to_thread(
                create_context,
                body["messages"],
                pipeline,
                send_event,
            )

            await queue.put({
                "type": "context_complete",
                "context": memory_context,
                "need_reasoning": need_reasoning,
            })

        except Exception as e:

            print(f"[Proxy] Context creation error: {e}")

            await queue.put({
                "type": "context_error",
                "error": str(e),
            })

    context_task = asyncio.create_task(
        build_context()
    )

    ###########################################################################
    # SSE generator
    ###########################################################################

    async def generator():

        client = None
        response = None

        try:

            ###################################################################
            # Wait for context while forwarding status events
            ###################################################################

            while True:

                event = await queue.get()

                event_type = event.get("type")

                ################################################################
                # Status event
                ################################################################

                if event_type == "status":

                    yield (
                        f"data: {json.dumps(event)}\n\n"
                    ).encode()

                    continue

                ################################################################
                # Context finished
                ################################################################

                if event_type == "context_complete":

                    memory_context = event["context"]
                    need_reasoning = event["need_reasoning"]

                    break

                ################################################################
                # Context failed
                ################################################################

                if event_type == "context_error":

                    error_event = {
                        "type": "status",
                        "stage": "planning",
                        "state": "failed",
                        "detail": event["error"],
                    }

                    yield (
                        f"data: {json.dumps(error_event)}\n\n"
                    ).encode()

                    return

            ###################################################################
            # Configure model request
            ###################################################################

            prepare_model_request(
                body,
                memory_context,
                need_reasoning,
                config["llama"]["name"],
            )

            ###################################################################
            # Connect to llama.cpp
            ###################################################################

            client = httpx.AsyncClient(timeout=None)

            request = client.build_request(
                "POST",
                f"{LLAMA_URL}/v1/chat/completions",
                json=body,
            )

            response = await client.send(
                request,
                stream=True,
            )

            ###################################################################
            # Thinking status
            ###################################################################

            if need_reasoning:

                thinking_event = {
                    "type": "status",
                    "stage": "thinking",
                    "state": "started",
                }

                yield (
                    f"data: {json.dumps(thinking_event)}\n\n"
                ).encode()

            ###################################################################
            # Forward llama.cpp stream
            ###################################################################

            async for chunk in response.aiter_bytes():

                yield chunk

            ###################################################################
            # Generation finished
            ###################################################################

            completed_event = {
                "type": "status",
                "stage": "generating",
                "state": "completed",
            }

            yield (
                f"data: {json.dumps(completed_event)}\n\n"
            ).encode()

        except asyncio.CancelledError:

            print("[Proxy] Client disconnected")

            raise

        except Exception as e:

            print(f"[Proxy] Streaming error: {e}")

            error_event = {
                "type": "status",
                "stage": "generating",
                "state": "failed",
                "detail": str(e),
            }

            yield (
                f"data: {json.dumps(error_event)}\n\n"
            ).encode()

        finally:

            ###################################################################
            # Cancel context task if still running
            ###################################################################

            if not context_task.done():

                context_task.cancel()

            ###################################################################
            # Close llama connection
            ###################################################################

            if response is not None:

                await response.aclose()

            if client is not None:

                await client.aclose()

    ###########################################################################
    # Return SSE response
    ###########################################################################

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


################################################################################
# Endpoints
################################################################################

@app.post("/v1/chat/completions")
async def chat(request: Request):

    body = await request.json()

    ###########################################################################
    # Model
    ###########################################################################

    model = config["llama"]["name"]

    print(f"[Proxy] Selected model: {model}")

    manager.ensure()

    ###########################################################################
    # Pipeline
    ###########################################################################

    pipeline_name = body.get("pipeline", "normal")

    pipeline = PIPELINES.get(
        pipeline_name,
        NORMAL_PIPELINE,
    )

    print(f"[Proxy] Selected pipeline: {pipeline_name}")

    ###########################################################################
    # Streaming request
    ###########################################################################

    if body.get("stream", False):

        return await stream_completion(
            body,
            pipeline,
        )

    ###########################################################################
    # Non-streaming request
    ###########################################################################

    memory_context, need_reasoning = create_context(
        body["messages"],
        pipeline,
    )

    prepare_model_request(
        body,
        memory_context,
        need_reasoning,
        config["llama"]["name"],
    )

    ###########################################################################
    # Forward
    ###########################################################################

    return await normal_completion(body)


################################################################################
# Models
################################################################################

@app.get("/v1/models")
def models():

    return {
        "object": "list",
        "data": [
            {
                "id": "jarvis",
                "object": "model",
                "owned_by": "jarvis",
            }
        ],
    }


################################################################################
# Health
################################################################################

@app.get("/health")
def health():

    return {
        "proxy": "ok",
        "running": manager.running(),
    }


################################################################################
# Root
################################################################################

@app.get("/")
def root():

    return {
        "status": "Jarvis Proxy Running"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config["chat_proxy"]["host"],
        port=config["chat_proxy"]["port"],
    )
