import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from notion_client import Client

from ..utils.config import settings
from ..utils.logger import logger


class NotionUploadError(Exception):
    """Raised when Notion upload fails"""
    pass


class NotionUploader:
    def __init__(self, config=None):
        self.config = config or settings.notion
        self.client = None
        
    def load_client(self):
        """Load Notion client"""
        if not self.config.enabled:
            return
            
        if self.client is not None:
            return
            
        if not settings.notion_token:
            raise NotionUploadError("Notion token not provided")
            
        logger.info("Initializing Notion client")
        self.client = Client(auth=settings.notion_token)
        
    def upload(
        self,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict:
        """Upload content to Notion"""
        if not self.config.enabled:
            logger.info("Notion upload is disabled")
            return {"uploaded": False, "reason": "disabled"}
            
        self.load_client()
        
        if not self.config.database_id:
            raise NotionUploadError("Notion database ID not provided")
            
        logger.info(f"Uploading to Notion: {title}")
        
        try:
            # Prepare page properties
            properties = self._prepare_properties(title, metadata)
            
            # Convert content to blocks
            blocks = self._content_to_blocks(content)
            
            if dry_run:
                logger.info("Dry run mode - not uploading to Notion")
                return {
                    "uploaded": False,
                    "dry_run": True,
                    "properties": properties,
                    "blocks": blocks[:5],  # Show first 5 blocks
                    "total_blocks": len(blocks),
                }
                
            # Create page
            page = self.client.pages.create(
                parent={"database_id": self.config.database_id},
                properties=properties,
                children=blocks[:100],  # Notion API limit
            )
            
            # Add remaining blocks if any
            if len(blocks) > 100:
                self._add_remaining_blocks(page["id"], blocks[100:])
                
            logger.info(f"Successfully uploaded to Notion: {page['url']}")
            
            return {
                "uploaded": True,
                "page_id": page["id"],
                "url": page["url"],
                "total_blocks": len(blocks),
            }
            
        except Exception as e:
            logger.error(f"Notion upload failed: {e}")
            raise NotionUploadError(f"Notion upload failed: {e}") from e
            
    def _prepare_properties(
        self, title: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Prepare page properties"""
        properties = {
            "Name": {"title": [{"text": {"content": title}}]},
        }
        
        # Add configured properties
        for prop in self.config.properties:
            if prop.name == "Name":
                continue
                
            if prop.type == "select":
                properties[prop.name] = {"select": {"name": prop.value}}
            elif prop.type == "number":
                value = prop.value
                if metadata and prop.name.lower() in metadata:
                    value = metadata[prop.name.lower()]
                properties[prop.name] = {"number": value}
            elif prop.type == "rich_text":
                value = prop.value or ""
                if metadata and prop.name.lower() in metadata:
                    value = str(metadata[prop.name.lower()])
                properties[prop.name] = {
                    "rich_text": [{"text": {"content": value[:2000]}}]
                }
            elif prop.type == "date":
                value = datetime.now().isoformat()
                if metadata and prop.name.lower() in metadata:
                    value = metadata[prop.name.lower()]
                properties[prop.name] = {"date": {"start": value}}
                
        return properties
        
    def _content_to_blocks(self, content: str) -> List[Dict]:
        """Convert content to Notion blocks"""
        if not self.config.markdown_to_blocks:
            # Simple paragraph blocks
            return self._split_into_blocks(content)
            
        # Parse markdown
        blocks = []
        lines = content.split("\n")
        current_block = []
        
        for line in lines:
            # Headers
            if line.startswith("# "):
                if current_block:
                    blocks.extend(self._create_paragraph_blocks("\n".join(current_block)))
                    current_block = []
                blocks.append(self._create_header_block(line[2:], 1))
                
            elif line.startswith("## "):
                if current_block:
                    blocks.extend(self._create_paragraph_blocks("\n".join(current_block)))
                    current_block = []
                blocks.append(self._create_header_block(line[3:], 2))
                
            elif line.startswith("### "):
                if current_block:
                    blocks.extend(self._create_paragraph_blocks("\n".join(current_block)))
                    current_block = []
                blocks.append(self._create_header_block(line[4:], 3))
                
            # Bullet points
            elif line.startswith("- ") or line.startswith("* "):
                if current_block and not any(
                    current_block[-1].startswith(prefix) for prefix in ["- ", "* "]
                ):
                    blocks.extend(self._create_paragraph_blocks("\n".join(current_block)))
                    current_block = []
                current_block.append(line)
                
            # Numbered lists
            elif re.match(r"^\d+\. ", line):
                if current_block and not re.match(r"^\d+\. ", current_block[-1]):
                    blocks.extend(self._create_paragraph_blocks("\n".join(current_block)))
                    current_block = []
                current_block.append(line)
                
            # Code blocks
            elif line.startswith("```"):
                if current_block:
                    blocks.extend(self._create_paragraph_blocks("\n".join(current_block)))
                    current_block = []
                # Find end of code block
                code_lines = []
                i = lines.index(line) + 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if code_lines:
                    blocks.append(self._create_code_block("\n".join(code_lines)))
                    
            # Regular text
            else:
                current_block.append(line)
                
        # Add remaining content
        if current_block:
            blocks.extend(self._create_paragraph_blocks("\n".join(current_block)))
            
        return blocks
        
    def _split_into_blocks(self, text: str) -> List[Dict]:
        """Split text into blocks respecting size limits"""
        blocks = []
        
        # Split by double newlines first
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            if len(para) <= self.config.max_block_length:
                if para.strip():
                    blocks.append(self._create_paragraph_block(para))
            else:
                # Split long paragraphs
                words = para.split()
                current = []
                current_len = 0
                
                for word in words:
                    if current_len + len(word) + 1 > self.config.max_block_length:
                        if current:
                            blocks.append(self._create_paragraph_block(" ".join(current)))
                        current = [word]
                        current_len = len(word)
                    else:
                        current.append(word)
                        current_len += len(word) + 1
                        
                if current:
                    blocks.append(self._create_paragraph_block(" ".join(current)))
                    
        return blocks
        
    def _create_paragraph_blocks(self, text: str) -> List[Dict]:
        """Create paragraph blocks from text"""
        if not text.strip():
            return []
            
        # Handle bullet points
        if text.startswith("- ") or text.startswith("* "):
            return [self._create_bullet_block(line[2:]) for line in text.split("\n")]
            
        # Handle numbered lists
        if re.match(r"^\d+\. ", text):
            return [
                self._create_numbered_block(re.sub(r"^\d+\. ", "", line))
                for line in text.split("\n")
            ]
            
        # Regular paragraphs
        return self._split_into_blocks(text)
        
    def _create_paragraph_block(self, text: str) -> Dict:
        """Create a paragraph block"""
        return {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
            },
        }
        
    def _create_header_block(self, text: str, level: int) -> Dict:
        """Create a header block"""
        header_type = ["heading_1", "heading_2", "heading_3"][level - 1]
        return {
            "type": header_type,
            header_type: {
                "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
            },
        }
        
    def _create_bullet_block(self, text: str) -> Dict:
        """Create a bullet list block"""
        return {
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
            },
        }
        
    def _create_numbered_block(self, text: str) -> Dict:
        """Create a numbered list block"""
        return {
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
            },
        }
        
    def _create_code_block(self, code: str) -> Dict:
        """Create a code block"""
        return {
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": code[:2000]}}],
                "language": "plain text",
            },
        }
        
    def _add_remaining_blocks(self, page_id: str, blocks: List[Dict]):
        """Add remaining blocks to page in batches"""
        batch_size = 100
        
        for i in range(0, len(blocks), batch_size):
            batch = blocks[i : i + batch_size]
            self.client.blocks.children.append(block_id=page_id, children=batch)
            logger.debug(f"Added blocks {i+1}-{i+len(batch)} to page")