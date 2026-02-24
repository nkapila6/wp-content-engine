# Shared agent state definitins in LangGraph
# 24/02/2026 01:06PM Nikhil Kapila

from typing import Type, TypedDict, Literal
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    # ddgs result
    ddgs_result: dict
    ddgs_result_summary:str

    # ripgrep queries
    rg_queries: list
    rg_results: dict
    rg_result_summary: str

    # planning agent
    
    # stitching agent

    # TODO: persona adaptation aspect to be looked at later
    
    # output

class Task(BaseModel):
    id: int
    title: str

    goal:str
