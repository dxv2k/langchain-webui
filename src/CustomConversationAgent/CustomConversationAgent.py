import re 
from typing import Optional, Tuple, List, Any
from langchain.agents import ConversationalAgent
from langchain.schema import AgentAction, AgentFinish

class CustomConversationalAgent(ConversationalAgent):
    
    def _is_include_thought(self, llm_output: str) -> bool:
        regex = r"Thought:(.*)"
        match = re.search(regex, llm_output)
        if match:
            return True
        return False

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        if f"{self.ai_prefix}:" in llm_output:
            return self.ai_prefix, llm_output.split(f"{self.ai_prefix}:")[-1].strip()

        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output)

        if match:
            action = match.group(1)
            action_input = match.group(2)
            return action.strip(), action_input.strip(" ").strip('"')
        elif not self._is_include_thought(llm_output=llm_output):
            # If result doesn't include thought, we will return them
            # This problem most likely due to prompt & hard to control 
            # therefore this is a duct & tape solution
            return self.ai_prefix, llm_output
        else:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                # {"output": "Agent stopped due to iteration limit or time limit."}, ""
                {"output": "Sorry, I don't know the answer to that question."}, ""
            )
        else:
            raise ValueError(
                f"Got unsupported early_stopping_method `{early_stopping_method}`"
            )
