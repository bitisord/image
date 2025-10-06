"""Pipeline for extracting product label data using a multimodal LLM and persisting
results into an H2 database.

The module exposes a ``MultimodalLabelIngestor`` class that performs the
following steps:

1. Encode an input image to base64 and send it to a multimodal model (for
   example GPT-4o) requesting a structured JSON response.
2. Normalise/validate the JSON payload.
3. Create the ``product_labels`` table if it does not already exist.
4. Insert or update the product row in the H2 database.

The code does not depend on a particular provider. As long as the provider
offers an HTTP API that can accept a base64 encoded image, you can implement
``_call_model`` accordingly. The default implementation targets the OpenAI
Responses API because it is widely available and works for both free-trial and
paid users.

Usage example (after installing the dependencies listed in requirements.txt)::

    export OPENAI_API_KEY="sk-..."
    python -m multimodal_pipeline \
        --image-path ./data/bottle.jpg \
        --jdbc-url "jdbc:h2:~/products" \
        --jdbc-driver ./drivers/h2-2.2.224.jar

The H2 JDBC driver can be downloaded from https://www.h2database.com.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

if importlib.util.find_spec("jaydebeapi") is None:
    raise ModuleNotFoundError(
        "No module named 'jaydebeapi'. Install dependencies with 'pip install -r requirements.txt' "
        "before running this script."
    )

import jaydebeapi
from openai import OpenAI

LOGGER = logging.getLogger(__name__)


@dataclass
class ProductRecord:
    """Normalised product information produced by the multimodal model."""

    sku: str
    product_name: str
    manufacturer: Optional[str]
    expiration_date: Optional[str]
    lot_number: Optional[str]
    additional_notes: Optional[str]

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ProductRecord":
        """Convert raw JSON payload from the LLM into a ``ProductRecord``.

        The model is instructed to emit a JSON object that contains the keys
        defined below. Defensive defaults are applied to keep the pipeline
        resilient to missing fields.
        """

        def _safe_get(key: str) -> Optional[str]:
            value = payload.get(key)
            if value is None:
                return None
            # ensure strings and strip whitespace
            return str(value).strip() or None

        sku = _safe_get("sku")
        if not sku:
            raise ValueError("The multimodal model did not return a SKU.")

        product_name = _safe_get("product_name") or "Unknown product"

        return cls(
            sku=sku,
            product_name=product_name,
            manufacturer=_safe_get("manufacturer"),
            expiration_date=_safe_get("expiration_date"),
            lot_number=_safe_get("lot_number"),
            additional_notes=_safe_get("additional_notes"),
        )


class MultimodalLabelIngestor:
    """High level orchestrator for the multimodal -> H2 ingestion pipeline."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        prompt_path: Optional[pathlib.Path] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "An OpenAI API key must be supplied via the constructor or the "
                "OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.prompt = (
            prompt_path.read_text(encoding="utf-8")
            if prompt_path and prompt_path.exists()
            else self._default_prompt()
        )
        self.client = OpenAI(api_key=self.api_key)

    @staticmethod
    def _default_prompt() -> str:
        return (
            "Tu es un agent qui lit l'étiquette d'un produit alimentaire. "
            "Analyse l'image fournie et renvoie un JSON strict avec les clés "
            "suivantes : sku, product_name, manufacturer, expiration_date, "
            "lot_number, additional_notes. Utilise une chaîne vide si tu ne "
            "vois pas l'information."
        )

    @staticmethod
    def encode_image(image_path: pathlib.Path) -> str:
        with image_path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")

    def _call_model(self, image_path: pathlib.Path) -> Dict[str, Any]:
        """Send the image to the multimodal model and parse the JSON payload."""

        LOGGER.info("Calling %s for %s", self.model, image_path)
        image_base64 = self.encode_image(image_path)

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self.prompt},
                        {"type": "input_image", "image_base64": image_base64},
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )

        # The Responses API returns a structured content list. The JSON object is
        # contained in the first item as text.
        try:
            raw_text = response.output[0].content[0].text
        except (AttributeError, IndexError, KeyError) as exc:
            raise RuntimeError(
                f"Unexpected schema returned by the multimodal model: {response!r}"
            ) from exc

        LOGGER.debug("Raw model output: %s", raw_text)
        return json.loads(raw_text)

    @staticmethod
    def _ensure_table(cursor: jaydebeapi.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS product_labels (
                sku VARCHAR(255) PRIMARY KEY,
                product_name VARCHAR(512) NOT NULL,
                manufacturer VARCHAR(255),
                expiration_date VARCHAR(64),
                lot_number VARCHAR(64),
                additional_notes CLOB
            )
            """
        )

    @staticmethod
    def _upsert_product(cursor: jaydebeapi.Cursor, record: ProductRecord) -> None:
        cursor.execute(
            """
            MERGE INTO product_labels AS target
            USING (SELECT ? AS sku, ? AS product_name, ? AS manufacturer, ? AS expiration_date,
                          ? AS lot_number, ? AS additional_notes) AS source
            ON target.sku = source.sku
            WHEN MATCHED THEN UPDATE SET
                product_name = source.product_name,
                manufacturer = source.manufacturer,
                expiration_date = source.expiration_date,
                lot_number = source.lot_number,
                additional_notes = source.additional_notes
            WHEN NOT MATCHED THEN INSERT
                (sku, product_name, manufacturer, expiration_date, lot_number, additional_notes)
            VALUES
                (source.sku, source.product_name, source.manufacturer, source.expiration_date,
                 source.lot_number, source.additional_notes)
            """,
            [
                record.sku,
                record.product_name,
                record.manufacturer,
                record.expiration_date,
                record.lot_number,
                record.additional_notes,
            ],
        )

    def ingest(
        self,
        image_path: pathlib.Path,
        jdbc_url: str,
        driver_path: pathlib.Path,
        username: str = "sa",
        password: str = "",
    ) -> ProductRecord:
        """Run the full pipeline for a single image."""

        payload = self._call_model(image_path)
        record = ProductRecord.from_payload(payload)

        connection = jaydebeapi.connect(
            "org.h2.Driver",
            jdbc_url,
            [username, password],
            str(driver_path),
        )
        try:
            cursor = connection.cursor()
            self._ensure_table(cursor)
            self._upsert_product(cursor, record)
            connection.commit()
        finally:
            connection.close()

        LOGGER.info("Ingested %s into product_labels", record.sku)
        return record


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", required=True, type=pathlib.Path)
    parser.add_argument("--jdbc-url", required=True)
    parser.add_argument("--jdbc-driver", required=True, type=pathlib.Path)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--prompt", type=pathlib.Path)
    parser.add_argument("--username", default="sa")
    parser.add_argument("--password", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    ingestor = MultimodalLabelIngestor(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=args.model,
        prompt_path=args.prompt,
    )

    record = ingestor.ingest(
        image_path=args.image_path,
        jdbc_url=args.jdbc_url,
        driver_path=args.jdbc_driver,
        username=args.username,
        password=args.password,
    )

    print(json.dumps(record.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
