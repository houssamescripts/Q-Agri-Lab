"""Responses API agent with bounded, explicit experiment tool calling."""
import json
import os

from app.agents.tools import AGENT_TOOLS, execute_tool, tool_output
from app.agents.systemprompt import SYSTEM_PROMPT


class ResearcherAgent:
    def __init__(self, client=None, model=None, max_tool_rounds=4):
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.max_tool_rounds = max_tool_rounds

    def run(self, message: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=message,
            tools=AGENT_TOOLS,
        )
        rounds = 0
        while rounds < self.max_tool_rounds:
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                return response.output_text
            outputs = []
            for call in calls:
                try:
                    arguments = json.loads(call.arguments)
                    result = execute_tool(call.name, arguments)
                    output = tool_output(result)
                except Exception as exc:
                    output = tool_output({"error": str(exc), "tool": call.name})
                outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
            response = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                previous_response_id=response.id,
                input=outputs,
                tools=AGENT_TOOLS,
            )
            rounds += 1
        return response.output_text or "Tool-call limit reached; summarize the results already returned."
