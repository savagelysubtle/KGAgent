"""
Custom parser for ChatGPT HTML export files.

ChatGPT exports contain JavaScript that renders conversations client-side.
This parser extracts the embedded JSON data and converts it to text.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.logging import logger


class GPTChatParser:
    """Parser for ChatGPT HTML export files."""

    def __init__(self, output_dir: str = "data/parsed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_gpt_export(self, html_content: str) -> bool:
        """Check if HTML file is a GPT chat export."""
        indicators = [
            "getConversationMessages",
            "jsonData",
            "conversation.mapping",
            "node.message.author.role",
        ]
        return any(indicator in html_content for indicator in indicators)

    def extract_json_data(self, html_content: str) -> Optional[List[Dict]]:
        """Extract the embedded JSON data from the HTML."""
        # GPT exports have JSON data in a script variable like:
        # var jsonData = [...];

        # Find the start of jsonData
        start_marker = "jsonData = ["
        start_idx = html_content.find(start_marker)

        if start_idx == -1:
            start_marker = "var jsonData = ["
            start_idx = html_content.find(start_marker)

        if start_idx == -1:
            logger.warning("Could not find jsonData in HTML content")
            return None

        # Find the actual start of the array
        array_start = html_content.find("[", start_idx)
        if array_start == -1:
            return None

        # Now we need to find the matching closing bracket
        # This is tricky because the JSON is huge - use bracket counting
        bracket_count = 0
        in_string = False
        escape_next = False
        array_end = -1

        for i in range(array_start, len(html_content)):
            char = html_content[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    array_end = i + 1
                    break

        if array_end == -1:
            logger.warning("Could not find end of JSON array")
            return None

        json_str = html_content[array_start:array_end]
        logger.info(f"Extracted JSON string of {len(json_str):,} bytes")

        try:
            data = json.loads(json_str)
            logger.info(f"Successfully parsed {len(data)} conversations")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # Try to find the error location
            logger.error(f"Error at position {e.pos}, line {e.lineno}")
            return None

    def extract_messages_from_conversation(self, conversation: Dict) -> List[Dict[str, Any]]:
        """Extract messages from a single conversation."""
        messages = []

        if 'mapping' not in conversation:
            return messages

        # Find the current node and traverse backwards
        current_node = conversation.get('current_node')
        mapping = conversation['mapping']

        # Build message chain
        message_chain = []
        while current_node and current_node in mapping:
            node = mapping[current_node]
            message = node.get('message')

            if message and message.get('content'):
                content = message['content']
                author = message.get('author', {}).get('role', 'unknown')

                # Skip system messages unless they're user system messages
                metadata = message.get('metadata', {})
                if author == 'system' and not metadata.get('is_user_system_message'):
                    current_node = node.get('parent')
                    continue

                # Extract text parts
                parts = content.get('parts', [])
                text_parts = []

                for part in parts:
                    if isinstance(part, str) and part.strip():
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        # Handle transcription
                        if part.get('content_type') == 'audio_transcription':
                            text_parts.append(f"[Audio Transcription]: {part.get('text', '')}")
                        # Handle other content types
                        elif 'text' in part:
                            text_parts.append(part['text'])

                if text_parts:
                    # Map author roles
                    if author in ('assistant', 'tool'):
                        display_author = 'ChatGPT'
                    elif author == 'user':
                        display_author = 'User'
                    elif author == 'system' and metadata.get('is_user_system_message'):
                        display_author = 'Custom User Info'
                    else:
                        display_author = author.title()

                    message_chain.append({
                        'author': display_author,
                        'text': '\n'.join(text_parts),
                        'timestamp': message.get('create_time'),
                    })

            current_node = node.get('parent')

        # Reverse to get chronological order
        return list(reversed(message_chain))

    def parse_to_text(self, html_content: str) -> str:
        """Parse GPT export HTML to plain text."""
        json_data = self.extract_json_data(html_content)

        if not json_data:
            logger.warning("Could not extract JSON data from GPT export")
            return ""

        text_parts = []

        for i, conversation in enumerate(json_data):
            title = conversation.get('title', f'Conversation {i + 1}')
            create_time = conversation.get('create_time')

            # Add conversation header
            text_parts.append(f"\n{'='*60}")
            text_parts.append(f"CONVERSATION: {title}")
            if create_time:
                try:
                    dt = datetime.fromtimestamp(create_time)
                    text_parts.append(f"Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    pass
            text_parts.append('='*60 + '\n')

            # Extract messages
            messages = self.extract_messages_from_conversation(conversation)

            for msg in messages:
                author = msg['author']
                text = msg['text']

                text_parts.append(f"\n[{author}]:")
                text_parts.append(text)
                text_parts.append('')  # Empty line between messages

        return '\n'.join(text_parts)

    def parse_file(self, file_path: str, job_id: str = "default") -> Optional[str]:
        """
        Parse a GPT chat export HTML file.

        Args:
            file_path: Path to the HTML file
            job_id: Job identifier

        Returns:
            Path to the parsed JSON file, or None if parsing failed
        """
        path_obj = Path(file_path)

        if not path_obj.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                html_content = f.read()

            if not self.is_gpt_export(html_content):
                logger.info(f"File {file_path} is not a GPT export")
                return None

            logger.info(f"Parsing GPT chat export: {file_path}")

            # Extract text
            text_content = self.parse_to_text(html_content)

            if not text_content:
                logger.warning(f"No content extracted from {file_path}")
                return None

            # Also extract raw JSON for metadata
            json_data = self.extract_json_data(html_content)

            # Create output structure
            output_data = {
                "main_text": text_content,
                "markdown": text_content,  # For compatibility with chunker
                "_metadata": {
                    "source_path": str(path_obj),
                    "source_type": "gpt_chat_export",
                    "conversation_count": len(json_data) if json_data else 0,
                    "parsed_at": datetime.utcnow().isoformat() + "Z",
                },
            }

            # Save to output directory
            job_dir = self.output_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"{path_obj.stem}.json"
            output_path = job_dir / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            # Log stats
            word_count = len(text_content.split())
            logger.info(f"Parsed GPT export: {len(json_data) if json_data else 0} conversations, ~{word_count} words")

            return str(output_path)

        except Exception as e:
            logger.error(f"Error parsing GPT export {file_path}: {e}")
            return None


# Singleton instance
_gpt_parser_instance: Optional[GPTChatParser] = None

def get_gpt_chat_parser() -> GPTChatParser:
    """Get or create GPT chat parser instance."""
    global _gpt_parser_instance
    if _gpt_parser_instance is None:
        _gpt_parser_instance = GPTChatParser()
    return _gpt_parser_instance

