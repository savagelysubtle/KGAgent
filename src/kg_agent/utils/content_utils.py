"""Content cleaning and processing utilities."""
from typing import Optional, List, Dict, Any
import re
from bs4 import BeautifulSoup
from ..core.logging import logger


def clean_html_content(html: str) -> str:
    """
    Clean HTML content by removing scripts, styles, and unwanted tags.

    Args:
        html: Raw HTML content

    Returns:
        Cleaned HTML content
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, str) and text.startswith('<!--')):
            comment.extract()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    except Exception as e:
        logger.error(f"Error cleaning HTML: {e}")
        return html


def extract_text_blocks(html: str, min_length: int = 50) -> List[str]:
    """
    Extract meaningful text blocks from HTML.

    Args:
        html: HTML content
        min_length: Minimum block length

    Returns:
        List of text blocks
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Find text blocks
        blocks = []
        for element in soup.find_all(['p', 'div', 'section', 'article', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = element.get_text().strip()
            if len(text) >= min_length:
                blocks.append(text)

        return blocks

    except Exception as e:
        logger.error(f"Error extracting text blocks: {e}")
        return []


def calculate_word_count(text: str) -> int:
    """
    Calculate word count in text.

    Args:
        text: Text content

    Returns:
        Number of words
    """
    if not text:
        return 0

    # Split on whitespace and filter out empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)


def extract_metadata_from_html(html: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML head section.

    Args:
        html: HTML content

    Returns:
        Dictionary of metadata
    """
    metadata = {}

    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()

        # Meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')

            if name and content:
                metadata[name.lower()] = content

        # Description
        if 'description' in metadata:
            metadata['description'] = metadata['description']
        elif 'og:description' in metadata:
            metadata['description'] = metadata['og:description']

        # Keywords
        if 'keywords' in metadata:
            keywords = metadata['keywords']
            if isinstance(keywords, str):
                metadata['keywords'] = [k.strip() for k in keywords.split(',')]

        # Author
        if 'author' in metadata:
            metadata['author'] = metadata['author']
        elif 'article:author' in metadata:
            metadata['author'] = metadata['article:author']

    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")

    return metadata


def detect_language(text: str) -> str:
    """
    Detect language of text content.

    Args:
        text: Text content

    Returns:
        Language code (default: 'en')
    """
    # Simple language detection based on common words
    # For production, use a proper language detection library

    text_lower = text.lower()

    # English indicators
    english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    english_count = sum(1 for word in english_words if word in text_lower)

    # German indicators
    german_words = ['der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'auf', 'at', 'zu', 'für', 'von', 'mit', 'bei']
    german_count = sum(1 for word in german_words if word in text_lower)

    if german_count > english_count:
        return 'de'
    else:
        return 'en'


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)

    # Replace spaces with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)

    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        name = name[:255-len(ext)-1] if ext else name[:255]
        sanitized = f"{name}.{ext}" if ext else name

    return sanitized
