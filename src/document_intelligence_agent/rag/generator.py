import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


class RAGGenerator:
    """Generate grounded answers from retrieved document evidence."""

    def __init__(
            self,
            model_name: str = "gpt-5.6-luna",
    ) -> None:
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(
            self,
            query: str,
            evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a grounded answer and return its source evidence."""

        context_parts = []

        for index, result in enumerate(evidence, start=1):
            metadata = result["metadata"]

            context_parts.append(
                f"""
Evidence {index}
Document: {metadata.get("document_id", "unknown")}
Page: {metadata.get("page_numbers", "unknown")}
Section: {metadata.get("section", "unknown")}

Text:
{result["text"]}
""".strip()
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
Answer the user's question using only the provided document evidence.

If the evidence does not contain enough information to answer the
question, say that the available document evidence is insufficient.

Do not invent facts or information that is not supported by the evidence.

Do not mention evidence numbers in the answer.

User question:
{query}

Document evidence:
{context}
""".strip()

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )

        sources = []

        for result in evidence:
            metadata = result["metadata"]

            sources.append(
                {
                    "document_id": metadata.get("document_id"),
                    "page_numbers": metadata.get("page_numbers"),
                    "section": metadata.get("section"),
                    "item_types": metadata.get("item_types"),
                    "provenance": metadata.get("provenance"),
                }
            )

        return {
            "answer": response.output_text,
            "sources": sources,
        }