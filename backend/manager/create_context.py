import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "Diary"))
from prompt_builder import build_memory_context



from Retrieval_planner import planner
from build_websearch_context import build_websearch_context


def create_context(messages, pipeline, send_event=None):
    """
    Build the complete retrieval context (memory + web) using the planner.

    Parameters
    ----------
    messages : list
        Conversation messages sent to the model.

    send_event : callable, optional
        Callback used to send status events to the frontend.

    top_k : int
        Number of diary entries candidates to retrieve per search query

    max_full_entries : int
            Number of full diary entries to keep for the final model context
    
    max_summarie : int
            Number of summarized diary entries to keep for the final model context

    

    Returns
    -------
    tuple
        (context, need_reasoning)
    """


    ###########################################################################
    # Status helper
    ###########################################################################

    def status(stage, state="started", detail=None):

        if send_event is not None:

            event = {
                "type": "status",
                "stage": stage,
                "state": state,
            }

            if detail is not None:
                event["detail"] = detail

            send_event(event)


    ###########################################################################
    # Find the latest user message
    ###########################################################################

    query = None

    def should_run_planner(messages):
        if not messages:
            return False
        return messages[-1].get("role") == "user"


    def get_planner_query(messages):
        if not should_run_planner(messages):
            return None
        return messages[-1].get("content", "")
    
    current_query = get_planner_query(messages)

    if not current_query:
        return "", False


    # Find the latest model/assistant answer
    def latest_assistant_answer(messages):
        for message in reversed(messages[:-1]):
            if message.get("role") == "assistant":
                return message.get("content", "")
        return None


    # Find the two most recent user queries before the current one
    def get_previous_user_query(messages):
        for message in reversed(messages[:-1]):
            if message.get("role") == "user":
                return message.get("content", "")
        return None


    latest_answer = latest_assistant_answer(messages)
    previous_user_query = get_previous_user_query(messages)

    # Construct the prompt for the planner
    if previous_user_query and latest_answer:
        prompt = f"""
        
        Current user query: {current_query},


        Latest assistant answer: {latest_answer},


        Previous user queries: {previous_user_query}"""
    else:
        prompt = current_query


    ###########################################################################
    # Planner
    ###########################################################################

    status("planning", "started")

    try :
        plan = planner(prompt, pipeline.planner_prompt)

        if isinstance(plan, str):
            plan = json.loads(plan)
    except Exception as e:

        status("planning", "failed",str(e))
        raise

    status("planning", "completed")

    diary_queries = plan.get("diary_queries", [])

    web_queries = plan.get("web_queries", [])

    need_reasoning = plan.get("need_reasoning", False)


    diary_queries = diary_queries[:pipeline.max_diary_queries]
    web_queries = web_queries[:pipeline.max_web_queries]

    ###########################################################################
    # Retrieve contexts
    ###########################################################################

    contexts = []

    if diary_queries and plan.get("need_diary"):

        status("diary", "started")
        print("ENTERING MEMORY_CONTEXT")

        try:
            memory_context = build_memory_context(
                diary_queries,
                pipeline.top_k,
                pipeline.max_full_entries,
                pipeline.max_summaries
            )

            if memory_context:
                contexts.append(memory_context)

        except Exception as e:
            status("diary", "failed", str(e))
            raise

        status("diary", "completed")


    if web_queries and plan.get("need_web"):

        status("web", "started")
        print("ENTERING WEBSEARCH_CONTEXT")

        try:
            web_context = build_websearch_context(
                web_queries,
                current_query
            )

            if web_context:
                contexts.append(web_context)
        except Exception as e:
            status("web", "failed", str(e))
            raise

        status("web", "completed")

    ###########################################################################

    return "\n\n".join(contexts), need_reasoning
