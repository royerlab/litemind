from typing import List, Optional, Dict, Any

from arbol import asection, aprint

from litemind.agent.agent import Agent
from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi, ModelFeatures
from litemind.apis.exceptions import APIError


class ReActAgent(Agent):
    """
    A ReAct (Reasoning and Acting) agent that implements reasoning through chain-of-thought
    and systematic action-observation loops.

    This agent extends the base Agent class to add structured reasoning capabilities while
    leveraging litemind's existing tool handling infrastructure.
    """

    def __init__(self,
                 api: BaseApi,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 toolset: Optional[ToolSet] = None,
                 name: str = "ReActAgent",
                 max_reasoning_steps: int = 5,
                 **kwargs):
        """Initialize the ReAct agent."""
        # Find suitable model if none provided
        if model is None:
            model = api.get_best_model([
                ModelFeatures.TextGeneration,
                ModelFeatures.StructuredTextGeneration
            ])
            if model is None:
                raise APIError("No suitable model found with required capabilities")

        # Verify model capabilities
        required_features = [
            ModelFeatures.TextGeneration,
            ModelFeatures.StructuredTextGeneration
        ]
        for feature in required_features:
            if not api.has_model_support_for(feature, model):
                raise APIError(f"Model {model} does not support {feature}")

        super().__init__(api=api,
                         model=model,
                         temperature=temperature,
                         toolset=toolset,
                         name=name,
                         **kwargs)

        self.max_reasoning_steps = max_reasoning_steps
        self._reset_state()

    def _reset_state(self):
        """Reset the agent's internal state."""
        self.reasoning_chain = []
        self._current_step = 0
        self._remaining_tokens = self.api.max_num_input_tokens(self.model)

    def _get_react_prompt(self) -> str:
        """Get the core ReAct prompt instructions."""
        return """You are a thoughtful AI assistant that carefully analyzes problems before acting.
Always follow this reasoning process:

1. THINK: Analyze the current situation, review previous steps, and plan what to do next
2. DECIDE: Choose to either:
   - Use an available tool to gather information
   - Provide a final answer if you have enough information
3. EXPLAIN: Always explain your reasoning before and after taking actions

Format your responses exactly like this:

THINK: [Your careful analysis of the situation]
DECIDE: [Either USE TOOL or FINAL ANSWER]
EXPLAIN: [Your explanation of what you learned or why you're providing this answer]

Important guidelines:
- Break down complex problems into steps
- Refer to previous steps in your reasoning
- Only provide a final answer when you're confident
- Always explain your thinking
"""

    def _create_system_message(self) -> Message:
        """Create a fresh system message with the current reasoning chain."""
        prompt = self._get_react_prompt()
        chain = self._format_reasoning_chain()

        if chain != "No previous steps.":
            prompt += "\n\nReview your previous steps to maintain context:\n" + chain

        return Message(role="system", text=prompt)

    def _format_reasoning_chain(self) -> str:
        """Format the reasoning chain for inclusion in prompts."""
        if not self.reasoning_chain:
            return "No previous steps."

        formatted = []
        for i, step in enumerate(self.reasoning_chain, 1):
            formatted.append(f"\nStep {i}:")
            for key, value in step.items():
                if value:
                    formatted.append(f"{key.upper()}: {value}")
        return "\n".join(formatted)

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse a response into its components with error handling."""
        components = {
            'think': '',
            'decide': '',
            'explain': ''
        }
        current = None

        try:
            for line in str(response).split('\n'):
                line = line.strip()
                if not line:
                    continue

                if line.upper().startswith('THINK:'):
                    current = 'think'
                    line = line[6:]
                elif line.upper().startswith('DECIDE:'):
                    current = 'decide'
                    line = line[7:]
                elif line.upper().startswith('EXPLAIN:'):
                    current = 'explain'
                    line = line[8:]
                elif current:  # Continue previous section
                    components[current] += ' ' + line
                    continue

                if current and line:
                    components[current] += line

            # Clean up and validate
            components = {k: v.strip() for k, v in components.items()}

            # Ensure we have minimum required components
            if not components['think'] or not components['decide']:
                raise ValueError("Missing required components")

            return components

        except Exception as e:
            # If parsing fails, try to salvage what we can
            if any(components.values()):
                return components

            # Last resort: treat entire response as decision
            return {
                'think': 'Analyzing response',
                'decide': str(response),
                'explain': 'Direct response'
            }

    def _should_continue_conversation(self, messages: List[Message]) -> bool:
        """Check if we can continue the conversation within token limits."""
        total_tokens = 0
        for message in messages:
            # Rough approximation of tokens (can be replaced with exact counting)
            total_tokens += self.api.count_tokens(str(message))  # Assuming ~4 chars per token

        return total_tokens < self._remaining_tokens

    def _is_final_answer(self, decision: str) -> bool:
        """Check if a decision string indicates a final answer."""
        decision = decision.strip().upper()
        return (decision.startswith('FINAL ANSWER') or
                decision.startswith('FINAL') or
                decision.startswith('ANSWER'))

    def __call__(self, *args, **kwargs) -> Message:
        """
        Process a query using the ReAct methodology.

        This implementation leverages litemind's existing tool handling while adding
        structured reasoning capabilities.
        """

        # Prepare call by extracting conversation, message, and text:
        self._prepare_call(args, kwargs)

        # Get last message in conversation
        last_message = self.conversation.get_last_message()

        with asection("Calling ReAct Agent"):
            with asection("With Message"):
                aprint(last_message)

            # Reset state for new query
            self._reset_state()

            # Create initial system message
            system_message = self._create_system_message()
            self.conversation.system_messages = [system_message]  # Replace any existing

            # Handle input using parent class (adds user message to conversation)
            super().__call__(*args, **kwargs)

            with asection("ReAct Reasoning Loop"):
                # Begin ReAct loop
                while self._current_step < self.max_reasoning_steps:

                    with asection(f"Reasoning Step {self._current_step + 1}"):

                        # Check token limit
                        if not self._should_continue_conversation(self.conversation.get_all_messages()):
                            aprint("Reached token limit!")
                            return Message(
                                role="assistant",
                                text="I apologize, but I've reached the token limit. Here's what I know so far: " +
                                     (self.reasoning_chain[-1].get('explain', '') if self.reasoning_chain else '')
                            )

                        try:
                            # Get model's next step
                            response = self.api.generate_text(
                                model_name=self.model,
                                messages=self.conversation.get_all_messages(),
                                temperature=self.temperature,
                                toolset=self.toolset
                            )

                            # Parse the structured response
                            step = self._parse_response(str(response))
                            self.reasoning_chain.append(step)
                            aprint("Step:", step)

                            # Update system message with new reasoning chain
                            self.conversation.system_messages = [self._create_system_message()]

                            # Check if we've reached a final answer
                            if self._is_final_answer(step['decide']):
                                explanation = step['explain'].strip()
                                decision_text = step['decide'].strip()

                                # Extract actual answer from decision text
                                answer_text = decision_text.split(':', 1)[1].strip() if ':' in decision_text else decision_text

                                if explanation:
                                    final_text = f"{answer_text}\n\nReasoning: {explanation}"
                                else:
                                    final_text = answer_text

                                with asection("Final Answer"):
                                    aprint(final_text)

                                return Message(role="assistant", text=final_text)

                            # Add response to conversation for tool handling
                            self.conversation.append(response)

                        except Exception as e:

                            aprint(f"Error in reasoning: {str(e)}")
                            # Handle any errors in the reasoning process
                            return Message(
                                role="assistant",
                                text=f"I encountered an error in my reasoning: {str(e)}. Here's what I know so far: " +
                                     (self.reasoning_chain[-1].get('explain', '') if self.reasoning_chain else '')
                            )

                        self._current_step += 1

            # If we exceed max steps, provide best effort response
            response= Message(
                role="assistant",
                text=(
                        "I apologize, but I've reached the maximum number of reasoning steps. "
                        "Here's what I've learned so far:\n\n" +
                        (self.reasoning_chain[-1].get('explain', '') if self.reasoning_chain else str(response))
                )
            )

            with asection("Final Answer"):
                aprint(response)

            # Add the response to the conversation
            self.conversation.append(response)

            return response