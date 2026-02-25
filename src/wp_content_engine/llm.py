# different llms for different use-cases
# 25/02/2026 01:34PM Nikhil Kapila

from langchain_openrouter import ChatOpenRouter 

# example from documentation
# https://docs.langchain.com/oss/python/integrations/chat/openrouter
# model = ChatOpenRouter(
#     model="anthropic/claude-sonnet-4.5",
#     temperature=0,
#     max_tokens=1024,
#     max_retries=2,
#     # other params...
# )

llm = ChatOpenRouter(
    model=""
)
