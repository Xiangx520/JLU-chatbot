"""将上传文件解析为正文，不涉及页面和知识库写入。"""

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.oxml.exceptions import InvalidXmlError
from docx.table import Table
from lxml.etree import XMLSyntaxError
from pypdf import PdfReader
from pypdf.errors import PyPdfError


class FileParseError(ValueError):
    """文件格式或正文不符合解析要求。"""


class FileParser:
    """无状态解析器，统一返回文本，保留原文件名由调用方负责。"""

    def parse_file(self, data: bytes, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".md", ".pdf", ".docx"}:
            raise FileParseError("不支持该文件格式，请上传 TXT、MD、PDF 或 DOCX 文件。")
        if not data:
            raise FileParseError("文件为空，请选择包含文本的文件。")

        if suffix in {".txt", ".md"}:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise FileParseError("无法读取文本，请将 TXT 或 MD 文件保存为 UTF-8 编码。") from exc
        elif suffix == ".pdf":
            text = self._parse_pdf(data)
        else:
            text = self._parse_docx(data)

        if not text.strip():
            raise FileParseError("未提取到有效正文，文件可能为空或仅含图片；暂不支持 OCR。")
        return text

    @staticmethod
    def _parse_pdf(data: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise FileParseError("暂不支持加密 PDF，请解密后重新上传。")
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            return "\n".join(pages)
        except PyPdfError as exc:
            raise FileParseError("无法解析 PDF，文件可能已损坏。") from exc

    @staticmethod
    def _parse_docx(data: bytes) -> str:
        try:
            document = Document(BytesIO(data))
            blocks = []
            for block in document.iter_inner_content():
                if isinstance(block, Table):
                    blocks.extend(
                        "\t".join(cell.text for cell in row.cells)
                        for row in block.rows
                    )
                else:
                    blocks.append(block.text)
            return "\n".join(blocks)
        except (BadZipFile, KeyError, XMLSyntaxError, InvalidXmlError, ValueError) as exc:
            raise FileParseError("无法解析 DOCX，文件可能已损坏或不是有效的 Word 文档。") from exc
