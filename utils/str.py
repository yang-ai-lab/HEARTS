from loguru import logger


def extract_combine_xml_blocks(text: str, tag: str = "execute") -> str:
    """
    Extract all code blocks enclosed in <tag>...</tag> tags from the given text
    and combine them into a single code string.
    """
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag in text and end_tag not in text:
        logger.warning(
            f"Found start tag <{tag}> but no end tag </{tag}>., auto-complete end tag at the end."
        )
        text += f"\n{end_tag}"
    code_blocks = []
    start = 0
    while True:
        start_idx = text.find(start_tag, start)
        if start_idx == -1:
            break
        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if end_idx == -1:
            break
        code_blocks.append(text[start_idx:end_idx].strip())
        start = end_idx + len(end_tag)
    return "\n".join(code_blocks)
