import dashscope
import gradio as gr
from dashscope import Generation

dashscope.api_key = "sk
** ** ** ** ** ** ** ** ** ** "


def chat_with_ai(message, history):


    """带上下文记忆的聊天函数"""
try:
# 将历史对话合并成上下文文本
conversation = "\n".join(
    [f"用户: {h[0]}\nAI: {h[1]}" for
     h in history if h]
)
prompt = conversation + "\n用户：" +
message
res = Generation.call(model="qwen
turbo
", prompt=prompt)
answer = res.output.text
return answer
except Exception as e:
return f"出错：{e}"
# 使用ChatInterface实现聊天模式
chatbot = gr.ChatInterface(
    fn=chat_with_ai,
    title="💬 通义千问 · 智能聊天助手",
    description="和AI聊聊天吧！它能写诗、答疑、
讲故事，还能帮你写代码哦～"
)
chatbot.launch()
# 改进版多轮上下文对话版
import dashscope
import gradio as gr
from dashscope import Generation

dashscope.api_key = "sk
** ** ** ** ** ** ** ** ** ** ** "


def chat_with_ai(message, history):


    try:
    messages = [
        {"role": "system", "content":
            "你是一个有帮助的中文助手。"}
    ]
for item in history:
# 情况 1：老版 history -> [user,
bot]
if isinstance(item, (list,
                     tuple)) and len(item) == 2:
    user_msg, bot_msg = item
messages.append({"role":
                     "user", "content": str(user_msg)})
messages.append({"role":
                     "assistant", "content": str(bot_msg)})
# 情况 2：新版 history ->
{"role": "...", "content": ...}
elif isinstance(item, dict) and
     "role" in item and "content" in item:
role = item.get("role",
"user")
content =
item.get("content", "")
# content 可能是 list（多模
态），这里简单拼成字符串
if isinstance(content,
list):
content = " ".join(
c.get("text", "")
if isinstance(c, dict) else str(c)
for c in content
)
elif not
isinstance(content, str):
content = str(content)
messages.append({"role":
                     role, "content": content})
# 兜底：任何奇怪结构，都当作用户一句
话
else:
messages.append({"role":
                     "user", "content": str(item)})
# 当前这一轮用户输入
messages.append({"role": "user",
                 "content": message})
res = Generation.call(
model = "qwen-turbo",
messages = messages,
result_format = "message",
)
return res.output.choices[0]
["message"]["content"]
except Exception as e:
return f"出错：{repr(e)}"
chatbot = gr.ChatInterface(
    fn=chat_with_ai,
    title="💬 通义千问 · 智能聊天助手",
    description="和AI聊聊天吧！它能写诗、答疑、
讲故事，还能帮你写代码哦～"
)
chatbot.launch()
