"""URL parsing and validation utilities."""
from typing import Optional, List, Set
from urllib.parse import urlparse, urljoin, ParseResult
import re
from ..core.logging import logger


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing fragments and query parameters.

    Args:
        url: URL to normalize

    Returns:
        Normalized URL
    """
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="", query="")
    return normalized.geturl()


def get_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL.

    Args:
        url: URL to extract domain from

    Returns:
        Domain name or None if invalid URL
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None


def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs belong to the same domain.

    Args:
        url1: First URL
        url2: Second URL

    Returns:
        True if same domain, False otherwise
    """
    domain1 = get_domain(url1)
    domain2 = get_domain(url2)
    return domain1 == domain2 and domain1 is not None


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract URLs from text using regex.

    Args:
        text: Text to extract URLs from

    Returns:
        List of extracted URLs
    """
    url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?'
    urls = re.findall(url_pattern, text)
    return list(set(urls))  # Remove duplicates


def resolve_url(base_url: str, relative_url: str) -> str:
    """
    Resolve a relative URL against a base URL.

    Args:
        base_url: Base URL
        relative_url: Relative URL to resolve

    Returns:
        Absolute URL
    """
    return urljoin(base_url, relative_url)


def filter_urls_by_pattern(urls: List[str], pattern: str) -> List[str]:
    """
    Filter URLs by regex pattern.

    Args:
        urls: List of URLs to filter
        pattern: Regex pattern to match

    Returns:
        Filtered list of URLs
    """
    try:
        regex = re.compile(pattern)
        return [url for url in urls if regex.match(url)]
    except re.error as e:
        logger.error(f"Invalid regex pattern '{pattern}': {e}")
        return []


def deduplicate_urls(urls: List[str]) -> List[str]:
    """
    Remove duplicate URLs while preserving order.

    Args:
        urls: List of URLs (possibly with duplicates)

    Returns:
        Deduplicated list of URLs
    """
    seen = set()
    result = []

    for url in urls:
        normalized = normalize_url(url)
        if normalized not in seen:
            seen.add(normalized)
            result.append(url)

    return result


def classify_url_type(url: str) -> str:
    """
    Classify URL type based on path and extension.

    Args:
        url: URL to classify

    Returns:
        URL type classification
    """
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Check for common file extensions
    if any(path.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.txt']):
        return 'document'
    elif any(path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp']):
        return 'image'
    elif any(path.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.wmv']):
        return 'video'
    elif any(path.endswith(ext) for ext in ['.mp3', '.wav', '.ogg']):
        return 'audio'
    elif path.endswith('.css'):
        return 'stylesheet'
    elif path.endswith('.js'):
        return 'javascript'
    elif '/api/' in path or '/api?' in url:
        return 'api'
    else:
        return 'page'
