import asyncio
import json
from typing import List

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    message_handler, CancellationToken, type_subscription, TopicId,
)
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage, AssistantMessage, FunctionExecutionResultMessage, FunctionExecutionResult,
)
from autogen_core.tools import Tool

from models.message import Message


@type_subscription(topic_type="tool_use_agent")
class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[Tool]) -> None:
        super().__init__("An agent with tools")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._tools = tool_schema

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> None:
        # Create a session of messages.
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            await self.publish_message(
                Message(content=create_result.content),
                topic_id=TopicId(type="user_topic_type", source="default"),
            )
            return None
        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )

        # Add the first model create result to the session.
        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
        )

        # Add the function execution results to the session.
        session.append(FunctionExecutionResultMessage(content=results))

        # Run the chat completion again to reflect on the history and function execution results.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(create_result.content, str)

        # Return the result as a message.
        await self.publish_message(
            Message(content=create_result.content),
            topic_id=TopicId(type="analyze_agent", source="default"),
        )

        return None

    async def _execute_tool_call(
            self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)


@type_subscription(topic_type="analyze_agent")
class AnalyzePostAgent(RoutedAgent):
    def __init__(self, description: str,
                 system_message: str, model_client: ChatCompletionClient) -> None:
        super().__init__(description)
        self._system_messages: List[LLMMessage] = [SystemMessage(content=system_message)]
        self._model_client = model_client

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> None:
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            await self.publish_message(
                Message(content=create_result.content),
                topic_id=TopicId(type="user_topic_type", source="default"),
            )
            return None

        assert isinstance(create_result.content, str)

        # Return the result as a message.
        await self.publish_message(
            Message(content=create_result.content),
            topic_id=TopicId(type="user_topic_type", source="default"),
        )

        return None


@type_subscription(topic_type="user_topic_type")
class UserAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A user agent that outputs the final result to the user.")

    @message_handler
    async def handle_final_copy(self, message: Message, ctx: MessageContext) -> None:
        print(message.content)
