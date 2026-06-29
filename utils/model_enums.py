from enum import Enum


class OpenAIModelNames(Enum):
    """name -> endpoint"""

    o3 = "o3"  # 2 / 8
    chatgpt_4o = "chatgpt-4o-latest"  # 5 / 15
    gpt_4_5 = "gpt-4.5-preview"  # 75 / 150
    gpt_4_1 = "gpt-4.1"  # 2 / 8
    o4_mini = "o4-mini"  # 1.1 / 4.4
    gpt_5 = "gpt-5"
    gpt_5_1 = "gpt-5.1-2025-11-13"  # 1.25 / 10
    gpt_5_mini = "gpt-5-mini"
    gpt_5_nano = "gpt-5-nano"
    gpt_4_1_mini = "gpt-4.1-mini"  # 0.4 / 1.6
    gpt_4o_mini = "gpt-4o-mini"  # 0.15 / 0.6
    gpt_4o = "gpt-4o"  # 2.5 / 10


class AWSBedrockModelNames(Enum):
    """name -> endpoint"""

    deepseek_v3 = "deepseek.v3-v1:0"
    deepseek_r1 = "us.deepseek.r1-v1:0"
    qwen_vl_235b = "qwen.qwen3-vl-235b-a22b"
    qwen_235b = "qwen.qwen3-235b-a22b-2507-v1:0"
    qwen_coder_480b = "qwen.qwen3-coder-480b-a35b-v1:0"
    moonshot_kimi_k2 = "moonshot.kimi-k2-thinking"
    gemma3_12b = "google.gemma-3-12b-it"
    claude_4_5_haiku = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    claude_4_5_sonnet = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    llama4_maverick = "us.meta.llama4-maverick-17b-instruct-v1:0"
    minimax_m2 = "minimax.minimax-m2"
    nemontron_nano_12b = "nvidia.nemotron-nano-12b-v2"
    mistral_large_3 = "mistral.mistral-large-3-675b-instruct"


class GeminiModelNames(Enum):
    """name -> endpoint"""

    gemini_2_5_flash = "gemini-2.5-flash"
    gemini_2_5_pro = "gemini-2.5-pro"
    gemini_3_flash_preview = "gemini-3-flash-preview"
    gemini_3_pro_preview = "gemini-3-pro-preview"
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
