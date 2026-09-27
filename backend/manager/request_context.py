"""Prepare a chat request with Jarvis retrieval context and model settings."""


def add_retrieval_context(messages, retrieval_context):
    if messages[0].get("role") == "system":
        existing_content = messages[0].get("content", "")
        if not existing_content.strip():
            messages[0]["content"] = retrieval_context
        else:
            messages[0]["content"] += "\n\n" + retrieval_context
    else:
        messages.insert(
            0,
            {"role": "system", "content": retrieval_context},
        )


def prepare_model_request(body, retrieval_context, need_reasoning, model_name):
    body["model"] = model_name
    body["chat_template_kwargs"] = {"enable_thinking": need_reasoning}
    add_retrieval_context(body["messages"], retrieval_context)
    return body
