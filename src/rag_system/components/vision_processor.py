"""Asynchronous vision processing component using GPT-4V."""

import asyncio
import base64
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import httpx

from src.rag_system.config import get_config
from src.rag_system.utils.logger import get_logger
from src.rag_system.utils.exceptions import VisionParsingError, APIRateLimitError
from src.rag_system.utils.retry_policy import RetryPolicy
from src.rag_system.utils.rate_limiter import AsyncRateLimiter
from src.rag_system.components.pdf_parser import DocumentElement

logger = get_logger(__name__)


class VisionProcessor:
    """Asynchronous GPT-4V vision processing component."""

    VISION_PROMPT = """You are an assistant that finds charts, graphs, or diagrams from an image and summarizes their information. 
    There could be multiple diagrams in one image, so explain each one of them separately. Ignore tables.
    
    The response must be a JSON in the following format: {"graphs": [<chart_1>, <chart_2>, <chart_3>]} 
    where <chart_1>, <chart_2>, and <chart_3> are descriptions of each graph found in the image.
    
    If no graph is found, return an empty list: {"graphs": []}.
    
    Do not append or add anything other than the JSON format response. Don't use markdown code blocks."""

    def __init__(self):
        """Initialize vision processor."""
        self.config = get_config()
        self.vision_config = self.config.vision_config
        self.rate_limiter = AsyncRateLimiter(
            requests_per_second=self.config.rate_limit_config.requests_per_second,
            burst_size=self.config.rate_limit_config.burst_size,
        )
        self.retry_policy = RetryPolicy(
            max_attempts=self.vision_config.retry_max_attempts,
            base_delay_seconds=1.0,
            backoff_factor=self.vision_config.retry_backoff_factor,
        )
        logger.info("Vision processor initialized", config=self.vision_config.dict())

    async def process_image(self, image_path: str, source_document: str) -> Optional[DocumentElement]:
        """
        Process a single image with GPT-4V.

        Args:
            image_path: Path to the image file.
            source_document: Source document name.

        Returns:
            Optional[DocumentElement]: Graph description element, or None if no graphs found.

        Raises:
            VisionParsingError: If processing fails.
        """
        try:
            await self.rate_limiter.acquire()

            logger.debug(f"Processing image: {image_path}")
            path = Path(image_path)

            if not path.exists():
                raise VisionParsingError(f"Image file not found: {image_path}", image_path=image_path)

            # Encode image to base64
            base64_image = self._encode_image(str(path))

            # Call GPT-4V with retry
            graphs = await self.retry_policy.execute_async(
                self._call_vision_api, base64_image, image_path
            )

            if graphs:
                description = f"{image_path}\n" + "\n".join(
                    f"{key}: {item[key]}" for key in item.keys()
                )
                element = DocumentElement(
                    type="graph",
                    text=description,
                    source_document=source_document,
                    metadata={"image_path": image_path, "graphs_found": len(graphs)},
                )
                logger.debug(f"Found {len(graphs)} graphs in {image_path}")
                return element
            else:
                logger.debug(f"No graphs found in {image_path}")
                return None

        except VisionParsingError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to process image: {image_path}",
                error=str(e),
                exc_info=True,
            )
            raise VisionParsingError(
                f"Vision processing failed: {str(e)}",
                image_path=image_path,
                details={"error": str(e)},
            )

    async def process_images_batch(self, image_paths: List[str], source_document: str) -> List[DocumentElement]:
        """
        Process multiple images concurrently with rate limiting.

        Args:
            image_paths: List of image file paths.
            source_document: Source document name.

        Returns:
            List[DocumentElement]: All graph description elements.
        """
        logger.info(f"Processing {len(image_paths)} images concurrently")

        tasks = [self.process_image(ip, source_document) for ip in image_paths]
        elements = []

        for coro in asyncio.as_completed(tasks):
            try:
                element = await coro
                if element:
                    elements.append(element)
            except VisionParsingError as e:
                logger.warning(f"Failed to process image", details=e.details)
                # Continue with other images

        logger.info(f"Processed all images, found {len(elements)} with graphs")
        return elements

    async def _call_vision_api(self, base64_image: str, image_path: str) -> List[Dict[str, Any]]:
        """
        Call GPT-4V API with image.

        Args:
            base64_image: Base64-encoded image.
            image_path: Original image path (for logging).

        Returns:
            List[Dict[str, Any]]: Graph descriptions.

        Raises:
            VisionParsingError: If API call fails.
            APIRateLimitError: If rate limited.
        """
        try:
            api_key = self.config.openai_api_key.get_secret_value()

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            payload = {
                "model": self.vision_config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": self.vision_config.max_tokens,
                "temperature": self.vision_config.temperature,
            }

            async with httpx.AsyncClient(timeout=self.vision_config.timeout_seconds) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    raise APIRateLimitError(
                        "Rate limited by OpenAI",
                        retry_after=retry_after,
                        api_name="OpenAI GPT-4V",
                    )

                if response.status_code != 200:
                    raise VisionParsingError(
                        f"API returned status {response.status_code}: {response.text}",
                        image_path=image_path,
                        status_code=response.status_code,
                    )

                response_json = response.json()
                graph_json_str = response_json["choices"][0]["message"]["content"]

                # Parse the JSON response
                try:
                    graph_data = json.loads(graph_json_str)
                    return graph_data.get("graphs", [])
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse vision API response as JSON",
                        response_text=graph_json_str[:200],
                        image_path=image_path,
                    )
                    return []

        except (APIRateLimitError, VisionParsingError):
            raise
        except Exception as e:
            logger.error(
                f"Vision API call failed: {str(e)}",
                image_path=image_path,
                exc_info=True,
            )
            raise VisionParsingError(
                f"Vision API call failed: {str(e)}",
                image_path=image_path,
                details={"error": str(e)},
            )

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """
        Encode image to base64.

        Args:
            image_path: Path to image file.

        Returns:
            str: Base64-encoded image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
