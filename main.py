import asyncio
import os

from autogen_core import SingleThreadedAgentRuntime, TopicId
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from agents.agents import ToolUseAgent, UserAgent, AnalyzePostAgent
from models.message import Message
from tools.twitter_tool import twitter_user_post_tool

load_dotenv()

model_client = OpenAIChatCompletionClient(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.GPT_4O,
        "structured_output": True,
    },
)


async def main():
    runtime = SingleThreadedAgentRuntime()

    await ToolUseAgent.register(
        runtime, "tool_use_agent",
        lambda: ToolUseAgent(
            model_client=model_client,
            tool_schema=[twitter_user_post_tool],
        ),
    )

    await AnalyzePostAgent.register(
        runtime,
        "analyze_agent",
        lambda: AnalyzePostAgent(
            description="Analyze the post content and return the sentiment score.",
            system_message="""
            你现在是一名财经专家，请对以下财经博主的发言进行分析，并给按我指定的格式返回分析结果。
            这是你需要分析的内容：{content}
            
            这是输出格式的说明：
              {{
                  "is_relevant": "是否与财经相关，且与美股市场或美债市场或科技股或半导体股或中国股票市场或香港股票市场或人民币兑美元汇率或中美关系相关。如果相关就返回1，如果不相关就返回0。只需要返回1或0这两个值之一即可",
                  "analytical_briefing": "分析简报"
              }}
            其中analytical_briefing的值是一个字符串，它是针对内容所做的分析简报，仅在is_relevant为1时会返回这个值。
            analytical_briefing的内容是markdown格式的，它需要符合下面的规范:
            
            原始正文，仅当需要分析的内容不是为中文时，这部分内容才会保留，否则这部分的内容为原始的正文
            翻译后的内容，仅当需要分析的内容为英文时，才会有这部分的内容。
            
            ## Brief Analysis
            分析结果。这部分会展示一个列表，列表中分别包含美股市场、美债市场、科技股、半导体股、中国股票市场、香港股票市场、人民币兑美元汇率、中美关系这8个选项。
            每个选项的值为分别为📈利多和📉利空。如果分析内容对于该选项没有影响，就不要针对这个选项返回任何内容。
            
            ## Summarize
            这部分需要用非常简明扼要的文字对分析结果进行总结，以及解释为什么在上面针对不同选项会得出不同的结论。
            """,
            model_client=model_client,
        ),
    )

    await UserAgent.register(
        runtime, "user_agent",
        lambda: UserAgent(),
    )

    runtime.start()

    await runtime.publish_message(
        Message(content="""
        请给我指定用户列表的一条最新推文, 列表中是他们对应的user_id:
        1. myfxtrader
        2. HAOHONG_CFA
        """),
        topic_id=TopicId(type="tool_use_agent", source="default"),
    )

    await runtime.stop_when_idle()
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
