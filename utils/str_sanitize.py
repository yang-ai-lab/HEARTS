def extract_md_img(exec_res: str):
    """
    Extracts all image URLs from a given Markdown string. Useful to remove images in result string returned from jupyter server execution. (tools.jupyter.JupyterKernel.execute)

    Parameters:
        out_str (str): The Markdown string to extract image URLs from.

    Returns:
        image_urls (list): A list of image URLs found in the Markdown string.
        clean_text (str): The cleaned Markdown string without image URLs.

    Raises:
        None
    """
    out_imgs = []
    clean_text = exec_res
    img_start_idx = clean_text.find("![image]")
    while img_start_idx != -1:
        img_end_idx = clean_text.find(")", img_start_idx)
        if img_end_idx == -1:
            break
        img_str = clean_text[img_start_idx : img_end_idx + 1]
        out_imgs.append(img_str)
        clean_text = clean_text[:img_start_idx] + clean_text[img_end_idx + 1 :]
        img_start_idx = clean_text.find("![image]")
    return out_imgs, clean_text


def md_img_to_url(markdown_img_str: str) -> str:
    """
    Extracts base64 data URL from a Markdown image string.
    Args:
        markdown_img_str (str): The Markdown image string to parse.
    Returns:
        str or None: The extracted image URL if found and valid, otherwise None.
    """
    try:
        # Find the starting position of the '!' which signals a Markdown image.
        # We need to ensure it's followed by '['
        bang_index = markdown_img_str.find("![")

        # If "![", is not found, or it's at the very end of the string
        # where there wouldn't be space for ']()', return None.
        if (
            bang_index == -1
            or bang_index + 1 >= len(markdown_img_str)
            or markdown_img_str[bang_index + 1] != "["
        ):
            return None

        # Find the closing bracket ']' after the alt text.
        # Start searching from after the '![', so it finds the correct closing bracket.
        closing_bracket_index = markdown_img_str.find("]", bang_index + 2)

        # If ']' is not found, or it's not followed by '(', it's not a valid image tag.
        if (
            closing_bracket_index == -1
            or closing_bracket_index + 1 >= len(markdown_img_str)
            or markdown_img_str[closing_bracket_index + 1] != "("
        ):
            return None

        # Find the opening parenthesis '(' which contains the URL.
        # It must immediately follow the closing bracket.
        opening_paren_index = closing_bracket_index + 1

        # Find the closing parenthesis ')' for the URL.
        # Start searching from after the opening parenthesis.
        closing_paren_index = markdown_img_str.find(")", opening_paren_index + 1)

        # If ')' is not found, or the URL part is empty (e.g., "!()"), it's not valid.
        if closing_paren_index == -1 or closing_paren_index == opening_paren_index + 1:
            return None

        # Extract the substring between the parentheses. This is our URL.
        url = markdown_img_str[opening_paren_index + 1 : closing_paren_index]

        # Basic check to ensure the extracted string actually looks like a URL.
        # This is a simple heuristic and not a full URL validation.
        if (
            url.startswith("http://")
            or url.startswith("https://")
            or url.startswith("/")
            or url.startswith("data")
        ):
            return url

        return None  # If it doesn't look like a URL, even if parentheses are found.
    except Exception:
        return None
