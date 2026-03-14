import asyncio
import base64
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llmcall.extract import (
    _build_image_content_block,
    _build_pdf_content_block,
    _detect_image_mime,
    _is_url,
    _source_to_data_uri,
    aextract_image,
    aextract_pdf,
    extract_image,
    extract_pdf,
)


class TestIsUrl:
    def test_http(self):
        assert _is_url("http://example.com/doc.pdf")

    def test_https(self):
        assert _is_url("https://example.com/doc.pdf")

    def test_local_path_string(self):
        assert not _is_url("/tmp/doc.pdf")

    def test_bytes(self):
        assert not _is_url(b"%PDF-1.4")

    def test_path_object(self):
        assert not _is_url(Path("/tmp/doc.pdf"))


class TestSourceToDataUri:
    def test_bytes_pdf(self, tmp_path):
        raw = b"%PDF-1.4 fake"
        uri = _source_to_data_uri(raw, "application/pdf")
        assert uri.startswith("data:application/pdf;base64,")
        encoded_part = uri.split(",", 1)[1]
        assert base64.standard_b64decode(encoded_part) == raw

    def test_file_path(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"hello pdf")
        uri = _source_to_data_uri(f, "application/pdf")
        assert uri.startswith("data:application/pdf;base64,")

    def test_string_path(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"hello pdf")
        uri = _source_to_data_uri(str(f), "application/pdf")
        assert uri.startswith("data:application/pdf;base64,")


class TestBuildPdfContentBlock:
    def test_url_uses_file_id(self):
        block = _build_pdf_content_block("https://example.com/doc.pdf")
        assert block["type"] == "file"
        assert block["file"]["file_id"] == "https://example.com/doc.pdf"
        assert block["file"]["format"] == "application/pdf"

    def test_bytes_uses_file_data(self):
        block = _build_pdf_content_block(b"%PDF-1.4 fake content")
        assert block["type"] == "file"
        assert block["file"]["file_data"].startswith("data:application/pdf;base64,")

    def test_local_path_uses_file_data(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        block = _build_pdf_content_block(f)
        assert block["type"] == "file"
        assert block["file"]["file_data"].startswith("data:application/pdf;base64,")


class TestBuildImageContentBlock:
    def test_url_uses_image_url(self):
        block = _build_image_content_block("https://example.com/img.png")
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/img.png"

    def test_bytes_uses_data_uri(self):
        block = _build_image_content_block(b"\x89PNG...", media_type="image/png")
        assert block["type"] == "image_url"
        assert block["image_url"]["url"].startswith("data:image/png;base64,")

    def test_local_path_auto_detects_png(self, tmp_path):
        f = tmp_path / "photo.png"
        f.write_bytes(b"\x89PNG fake")
        block = _build_image_content_block(f)
        assert block["image_url"]["url"].startswith("data:image/png;base64,")

    def test_explicit_media_type_overrides_detection(self, tmp_path):
        f = tmp_path / "photo.bin"
        f.write_bytes(b"fake image data")
        block = _build_image_content_block(f, media_type="image/webp")
        assert block["image_url"]["url"].startswith("data:image/webp;base64,")


class TestDetectImageMime:
    def test_png_path(self):
        assert _detect_image_mime(Path("/tmp/photo.png")) == "image/png"

    def test_jpeg_path(self):
        assert _detect_image_mime(Path("/tmp/photo.jpg")) == "image/jpeg"

    def test_gif_string(self):
        assert _detect_image_mime("/tmp/anim.gif") == "image/gif"

    def test_bytes_fallback(self):
        assert _detect_image_mime(b"\xff\xd8\xff") == "image/jpeg"

    def test_unknown_extension_fallback(self):
        assert _detect_image_mime("/tmp/image.unknownext") == "image/jpeg"


class SimpleSchema(BaseModel):
    title: str
    summary: Optional[str] = None


def _make_mock_response(schema: BaseModel):
    import json

    msg = MagicMock()
    msg.content = json.dumps({"title": "Test", "summary": "A summary"})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestExtractPdfValidation:
    def test_raises_when_model_does_not_support_pdf(self, monkeypatch):
        monkeypatch.setenv("LLMCALL_API_KEY", "test-key")
        monkeypatch.setenv("LLMCALL_MODEL", "openai/gpt-3.5-turbo")

        with patch("llmcall.extract.supports_pdf_input", return_value=False):
            with pytest.raises(ValueError, match="PDF input is not supported"):
                extract_pdf("https://example.com/doc.pdf", SimpleSchema)

    def test_async_raises_when_model_does_not_support_pdf(self, monkeypatch):
        monkeypatch.setenv("LLMCALL_API_KEY", "test-key")
        monkeypatch.setenv("LLMCALL_MODEL", "openai/gpt-3.5-turbo")

        with patch("llmcall.extract.supports_pdf_input", return_value=False):
            with pytest.raises(ValueError, match="PDF input is not supported"):
                asyncio.run(aextract_pdf("https://example.com/doc.pdf", SimpleSchema))

    def test_succeeds_with_supported_model(self, monkeypatch):
        monkeypatch.setenv("LLMCALL_API_KEY", "test-key")
        monkeypatch.setenv("LLMCALL_MODEL", "openai/gpt-4o")

        with patch("llmcall.extract.supports_pdf_input", return_value=True), patch(
            "llmcall.extract.completion", return_value=_make_mock_response(SimpleSchema)
        ):
            result = extract_pdf("https://example.com/doc.pdf", SimpleSchema)
        assert result.title == "Test"

    def test_async_succeeds_with_supported_model(self, monkeypatch):
        monkeypatch.setenv("LLMCALL_API_KEY", "test-key")
        monkeypatch.setenv("LLMCALL_MODEL", "openai/gpt-4o")

        async def _run():
            with patch("llmcall.extract.supports_pdf_input", return_value=True), patch(
                "llmcall.extract.acompletion",
                new_callable=AsyncMock,
                return_value=_make_mock_response(SimpleSchema),
            ):
                return await aextract_pdf("https://example.com/doc.pdf", SimpleSchema)

        result = asyncio.run(_run())
        assert result.title == "Test"


class TestExtractImageValidation:
    def test_raises_when_model_does_not_support_vision(self, monkeypatch):
        monkeypatch.setenv("LLMCALL_API_KEY", "test-key")
        monkeypatch.setenv("LLMCALL_MODEL", "openai/gpt-3.5-turbo")

        with patch("llmcall.extract.supports_vision", return_value=False):
            with pytest.raises(ValueError, match="Vision/image input is not supported"):
                extract_image("https://example.com/img.png", SimpleSchema)

    def test_async_raises_when_model_does_not_support_vision(self, monkeypatch):
        monkeypatch.setenv("LLMCALL_API_KEY", "test-key")
        monkeypatch.setenv("LLMCALL_MODEL", "openai/gpt-3.5-turbo")

        with patch("llmcall.extract.supports_vision", return_value=False):
            with pytest.raises(ValueError, match="Vision/image input is not supported"):
                asyncio.run(aextract_image("https://example.com/img.png", SimpleSchema))

    def test_succeeds_with_vision_model(self, monkeypatch):
        monkeypatch.setenv("LLMCALL_API_KEY", "test-key")
        monkeypatch.setenv("LLMCALL_MODEL", "openai/gpt-4o")

        with patch("llmcall.extract.supports_vision", return_value=True), patch(
            "llmcall.extract.completion", return_value=_make_mock_response(SimpleSchema)
        ):
            result = extract_image("https://example.com/img.png", SimpleSchema)
        assert result.title == "Test"

    def test_async_succeeds_with_vision_model(self, monkeypatch):
        monkeypatch.setenv("LLMCALL_API_KEY", "test-key")
        monkeypatch.setenv("LLMCALL_MODEL", "openai/gpt-4o")

        async def _run():
            with patch("llmcall.extract.supports_vision", return_value=True), patch(
                "llmcall.extract.acompletion",
                new_callable=AsyncMock,
                return_value=_make_mock_response(SimpleSchema),
            ):
                return await aextract_image("https://example.com/img.png", SimpleSchema)

        result = asyncio.run(_run())
        assert result.title == "Test"
