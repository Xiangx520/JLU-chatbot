from pydantic import BaseModel, Field





# 本轮网页搜索返回的来源
class Reference(BaseModel):
    title: str = Field(description="网页标题")
    url: str = Field(description="网页链接")

# Agent 返回给聊天页面的回答；未联网时不展示来源列表。
class AnswerInfo(BaseModel):
    answer: str = Field(description="给用户的最终回答")
    reference: list[Reference] = Field(
        default_factory=list, description="本轮网页搜索返回的来源，不代表全部被回答引用"
    )
