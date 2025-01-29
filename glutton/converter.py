"""Core converter functionality for Glutton."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypedDict, Union

import yaml


class ConversionResult:
    """Represents the result of a conversion operation."""

    def __init__(
        self, input_file: str, output_file: Optional[str], text_content: str
    ) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.text_content = text_content


class OcrConfig(TypedDict, total=False):
    """OCR configuration options."""
    engine: str
    languages: List[str]


class DirectoryConfig(TypedDict):
    """Directory configuration options."""
    mode: str  # input_dir, custom_dir, working_dir
    default_path: str
    create_if_missing: bool

class OutputConfig(TypedDict):
    """Output configuration options."""
    default_pattern: str
    default_extension: str
    directory: DirectoryConfig

class Config(TypedDict):
    """Type definition for configuration options."""
    output: OutputConfig
    conversion: dict[str, str]
    ocr: OcrConfig


class BaseBackend(ABC):
    """Base class for conversion backends."""

    @abstractmethod
    def convert_to_markdown(self, input_file: str) -> str:
        """Convert input file to markdown."""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of file extensions this backend supports."""
        raise NotImplementedError


class MarkItDownBackend(BaseBackend):
    """MarkItDown-based conversion backend."""

    def __init__(self) -> None:
        try:
            from markitdown import MarkItDown
            self._converter = MarkItDown()
        except ImportError:
            raise ImportError(
                "markitdown package is required for this backend. "
                "Please install it before using."
            )

    def convert_to_markdown(self, input_file: str) -> str:
        result = self._converter.convert(input_file)
        return result.text_content

    @property
    def supported_formats(self) -> List[str]:
        return [".xlsx", ".xls", ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".html", ".txt", ".zip"]


class DoclingBackend(BaseBackend):
    """Docling-based conversion backend with advanced configuration."""

    def __init__(self, ocr_config: Optional[OcrConfig] = None) -> None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                EasyOcrOptions,
                OcrMacOptions,
                PdfPipelineOptions,
                RapidOcrOptions,
                TableFormerMode,
                TableStructureOptions,
            )
            from docling.document_converter import (
                DocumentConverter,
                FormatOption,
                PdfFormatOption,
            )

            # Get OCR engine and languages from config or use defaults
            ocr_engine = ocr_config.get("engine", "easyocr") if ocr_config else "easyocr"
            languages = ocr_config.get("languages", ["en"]) if ocr_config else ["en"]
            
            # Configure OCR options based on selected engine
            if ocr_engine == "easyocr":
                ocr_options = EasyOcrOptions(
                    kind="easyocr",
                    lang=languages,
                )
            elif ocr_engine == "rapidocr":
                ocr_options = RapidOcrOptions(
                    kind="rapidocr",
                    lang=languages,
                )
            elif ocr_engine == "ocrmac":
                ocr_options = OcrMacOptions(
                    kind="ocrmac",
                    lang=languages,
                )
            else:
                # Default to EasyOCR
                ocr_options = EasyOcrOptions(
                    kind="easyocr",
                    lang=languages,
                )
            
            # Configure table structure options
            table_options = TableStructureOptions(
                mode=TableFormerMode.ACCURATE,
                do_cell_matching=False,
            )
            
            # Configure pipeline options for best quality
            pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                ocr_options=ocr_options,
                table_structure_options=table_options,
            )
            
            # Create format options
            format_options: Dict[InputFormat, FormatOption] = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
            
            # Initialize converter with format-specific options
            self._converter = DocumentConverter(format_options=format_options)
            
        except ImportError:
            raise ImportError(
                "docling package is required for this backend. "
                "Please install it before using."
            )

    def convert_to_markdown(self, input_file: str) -> str:
        # Convert using docling with advanced options
        result = self._converter.convert(input_file)
        return result.document.export_to_markdown()

    @property
    def supported_formats(self) -> List[str]:
        """List of file extensions supported by docling."""
        return [
            # Document formats
            ".pdf",     # PDF documents
            ".doc",     # Microsoft Word
            ".docx",    # Microsoft Word
            ".rtf",     # Rich Text Format
            ".odt",     # OpenDocument Text
            ".txt",     # Plain text
            
            # Presentation formats
            ".ppt",     # Microsoft PowerPoint
            ".pptx",    # Microsoft PowerPoint
            ".odp",     # OpenDocument Presentation
            
            # Web formats
            ".html",    # HTML documents
            ".htm",     # HTML documents
            ".mht",     # MHTML web archive
            ".mhtml",   # MHTML web archive
            
            # Image formats
            ".png",     # PNG images
            ".jpg",     # JPEG images
            ".jpeg",    # JPEG images
            ".tiff",    # TIFF images
            ".bmp",     # Bitmap images
            
            # Archive formats
            ".zip",     # ZIP archives
            ".7z",      # 7-Zip archives
            ".rar",     # RAR archives
            ".tar",     # TAR archives
            ".gz",      # Gzip archives
            
            # E-book formats
            ".epub",    # EPUB e-books
            ".mobi",    # Mobipocket e-books
            ".azw",     # Kindle format
            
            # Other formats
            ".xml",     # XML documents
            ".json",    # JSON documents
            ".csv",     # CSV spreadsheets
            ".tsv",     # TSV spreadsheets
        ]


class GluttonConverter:
    """Main converter class supporting multiple conversion backends."""

    DEFAULT_CONFIG: Config = {
        "output": {
            "default_pattern": "{name}_md.{ext}",
            "default_extension": "md",
            "directory": {
                "mode": "input_dir",
                "default_path": "{home}/converted",
                "create_if_missing": True
            }
        },
        "conversion": {
            "encoding": "utf-8",
            "table_format": "github",
            "default_backend": "markitdown",
        },
        "ocr": {
            "engine": "easyocr",  # Options: easyocr, rapidocr, ocrmac
            "languages": ["en"],  # List of language codes
        },
    }

    def __init__(self, config_file: Optional[Union[str, Path]] = None) -> None:
        """Initialize converter with optional configuration file."""
        self.config = self.DEFAULT_CONFIG.copy()
        if config_file:
            self._load_config(config_file)

        # Initialize available backends
        self._backends: Dict[str, Type[BaseBackend]] = {
            "markitdown": MarkItDownBackend,
            "docling": DoclingBackend,
        }

    def _load_config(self, config_file: Union[str, Path]) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self.config["output"].update(user_config.get("output", {}))
                    self.config["conversion"].update(user_config.get("conversion", {}))
                    if "ocr" in user_config:
                        self.config["ocr"].update(user_config["ocr"])

    def _generate_output_name(
        self, input_file: str, output: Optional[str] = None
    ) -> Optional[str]:
        """Generate output filename based on pattern or explicit name."""
        if output:
            return output

        input_path = Path(input_file)
        pattern = self.config["output"]["default_pattern"]
        ext = self.config["output"]["default_extension"]
        dir_config = self.config["output"]["directory"]
        
        # Generate the filename part
        now = datetime.now()
        filename = pattern.format(
            name=input_path.stem,
            ext=ext,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H%M%S"),
        )

        # Determine output directory based on mode
        if dir_config["mode"] == "input_dir":
            output_dir = input_path.parent
        elif dir_config["mode"] == "custom_dir":
            # Replace placeholders in the path
            custom_path = dir_config["default_path"].format(
                home=str(Path.home()),
                input_dir=str(input_path.parent),
                date=now.strftime("%Y-%m-%d")
            )
            output_dir = Path(custom_path)
        else:  # working_dir
            output_dir = Path.cwd()

        # Create directory if needed
        if dir_config.get("create_if_missing", True):
            output_dir.mkdir(parents=True, exist_ok=True)

        # Combine directory and filename
        return str(output_dir / filename)

    def _get_backend(self, file_path: str, preferred_backend: Optional[str] = None) -> BaseBackend:
        """Get appropriate backend for the file type with fallback support."""
        ext = Path(file_path).suffix.lower()
        if not ext:
            raise ValueError(f"Cannot determine file type for: {file_path}")

        available_backends = []
        
        # Try preferred backend first
        if preferred_backend:
            try:
                backend_class = self._backends[preferred_backend]
                if issubclass(backend_class, DoclingBackend):
                    backend = backend_class(self.config.get("ocr"))
                else:
                    backend = backend_class()
                if ext in backend.supported_formats:
                    return backend
                available_backends.append((preferred_backend, backend.supported_formats))
            except (KeyError, ImportError):
                pass  # Fallback to other backends

        # Try default backend
        default_backend = self.config["conversion"].get("default_backend", "markitdown")
        if default_backend != preferred_backend:
            try:
                backend_class = self._backends[default_backend]
                if issubclass(backend_class, DoclingBackend):
                    backend = backend_class(self.config.get("ocr"))
                else:
                    backend = backend_class()
                if ext in backend.supported_formats:
                    return backend
                available_backends.append((default_backend, backend.supported_formats))
            except (KeyError, ImportError):
                pass

        # Try all other backends
        for backend_name, backend_class in self._backends.items():
            if backend_name in (preferred_backend, default_backend):
                continue
            try:
                if issubclass(backend_class, DoclingBackend):
                    backend = backend_class(self.config.get("ocr"))
                else:
                    backend = backend_class()
                if ext in backend.supported_formats:
                    return backend
                available_backends.append((backend_name, backend.supported_formats))
            except ImportError:
                continue

        # If we get here, no backend supports this format
        backend_info = "\n".join(
            f"  - {name}: {', '.join(formats)}"
            for name, formats in available_backends
        )
        raise ValueError(
            f"No compatible backend found for {ext} files.\n"
            f"Available backends and their supported formats:\n{backend_info}"
        )

    def convert(
        self, input_file: str, output: Optional[str] = None, backend: Optional[str] = None
    ) -> ConversionResult:
        """Convert a single file to markdown using the appropriate backend."""
        converter = self._get_backend(input_file, backend)
        md_content = converter.convert_to_markdown(input_file)
        
        output_file = self._generate_output_name(input_file, output)
        if output_file:
            with open(output_file, "w", encoding=self.config["conversion"]["encoding"]) as f:
                f.write(md_content)

        return ConversionResult(input_file, output_file, md_content)

    def convert_many(
        self, input_files: List[str], output_pattern: Optional[str] = None, backend: Optional[str] = None
    ) -> List[ConversionResult]:
        """Convert multiple files to markdown."""
        results = []
        for input_file in input_files:
            output = (
                output_pattern.format(
                    name=Path(input_file).stem,
                    ext=self.config["output"]["default_extension"],
                    date=datetime.now().strftime("%Y-%m-%d"),
                    time=datetime.now().strftime("%H%M%S"),
                )
                if output_pattern
                else None
            )
            results.append(self.convert(input_file, output, backend))
        return results

    @property
    def available_backends(self) -> List[str]:
        """List all available conversion backends."""
        return sorted(self._backends.keys())