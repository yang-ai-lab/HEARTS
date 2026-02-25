from enum import Enum


class OpenAIModelNames(Enum):
    """name -> endpoint"""

    gpt_5_mini = "gpt-5-mini"
    gpt_4_1_mini = "gpt-4.1-mini"


class AWSBedrockModelNames(Enum):
    """name -> endpoint"""

    deepseek_v3 = "deepseek.v3-v1:0"
    qwen_235b = "qwen.qwen3-235b-a22b-2507-v1:0"
    qwen_coder_480b = "qwen.qwen3-coder-480b-a35b-v1:0"
    moonshot_kimi_k2 = "moonshot.kimi-k2-thinking"
    claude_4_5_haiku = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    llama4_maverick = "us.meta.llama4-maverick-17b-instruct-v1:0"
    minimax_m2 = "minimax.minimax-m2"
    nemontron_nano_12b = "nvidia.nemotron-nano-12b-v2"


class GeminiModelNames(Enum):
    """name -> endpoint"""

    gemini_2_5_flash = "gemini-2.5-flash"
    gemini_2_5_pro = "gemini-2.5-pro"
    glm_4_7 = "zai-org/glm-4.7-maas"

class XAIModelNames(Enum):
    """name -> endpoint"""

    grok_4_1_fast = "grok-4-1-fast-reasoning"


TEXT_ONLY_MODELS = frozenset(
    [
        AWSBedrockModelNames.deepseek_v3.value,
        AWSBedrockModelNames.qwen_235b.value,
        AWSBedrockModelNames.qwen_coder_480b.value,
        AWSBedrockModelNames.moonshot_kimi_k2.value,
        AWSBedrockModelNames.minimax_m2.value,
        GeminiModelNames.glm_4_7.value,
    ]
)


ModelNames = Enum(
    "ModelNames",
    {
        **{model.name: model.value for model in OpenAIModelNames},
        **{model.name: model.value for model in AWSBedrockModelNames},
        **{model.name: model.value for model in GeminiModelNames},
        **{model.name: model.value for model in XAIModelNames},
    },
)


__all__ = ["OpenAIModelNames", "AWSBedrockModelNames", "GeminiModelNames", "XAIModelNames", "ModelNames"]
