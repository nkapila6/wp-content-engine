# node definitions for different nodes
# 24/02/2026 01:06PM Nikhil Kapila

from ddgs import DDGS
from utils.fetch import fetch_all_content

# ddgs web search
def ddgs_search(queries:list, num_results:int=10)->dict:
    results = {}
    ddgs = DDGS()
    
    for query in queries:
        res = ddgs.text(query, max_results=num_results, backend="google")
        # returns a List[Dict] of title, href, body
        results[query] = fetch_all_content(res)

    return results

# ripgrep fetch from kb
def generate_ripgrep_queries(start_prompt:str)->list:
    # generates ripgrep queries through LLM and returns a list
    return []

def exec_ripgrep_queries(queries:list)->dict:
    # executes ripgrep queries and spits out a dict
    return {}

# condensing node for planner
def condenser()->str:
    # condenses both ripgrep kb and search queries as context to plan blog post
    return ""

# planning node
def planner()->str: #need to change this to Plan later
    # outputs plan that LLM swarm need to work on
    return ""

def llm_swarm_write()->str:
    # each LLM writes one section as dictated by planner
    return ""

# stitching node
def stitcher()->str:
    # ensures that LLM swarm written work flows well and stitches together
    return ""

# styler based on persona and example
def styler()->str:
    # styles blog post based on given persona and example
    return ""
