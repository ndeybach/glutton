"""Core converter functionality for Glutton."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import yaml
from typing_extensions import TypedDict


class ConversionResult:
    """Represents the result of a conversion operation."""

    def __init__(
        self, input_file: str, output_file: Optional[str], text_content: str
    ) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.text_content = text_content


class Config(TypedDict):
    """Type definition for configuration options."""

    output: dict[str, str]
    conversion: dict[str, str]


class GluttonConverter:
    """Main converter class supporting multiple file formats.
    
    This class acts as a wrapper around various conversion backends,
    currently using markitdown as the primary conversion engine.
    Future implementations may support additional backends.
    """

    DEFAULT_CONFIG: Config = {
        "output": {
            "default_pattern": "{name}_md.{ext}",
            "default_extension": "md",
        },
        "conversion": {
            "encoding": "utf-8",
            "table_format": "github",
        },
    }

    def __init__(self, config_file: Optional[Union[str, Path]] = None) -> None:
        """Initialize converter with optional configuration file."""
        self.config = self.DEFAULT_CONFIG.copy()
        if config_file:
            self._load_config(config_file)
        
        try:
            from markitdown import MarkItDown
            self._converter = MarkItDown()
        except ImportError:
            raise ImportError(
                "markitdown package is required. "
                "Please install it before using this converter."
            )

    def _load_config(self, config_file: Union[str, Path]) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self.config["output"].update(user_config.get("output", {}))
                    self.config["conversion"].update(user_config.get("conversion", {}))

    def _generate_output_name(
        self, input_file: str, output: Optional[str] = None
    ) -> Optional[str]:
        """Generate output filename based on pattern or explicit name."""
        if output:
            return output

        input_path = Path(input_file)
        pattern = self.config["output"]["default_pattern"]
        ext = self.config["output"]["default_extension"]
        
        now = datetime.now()
        return pattern.format(
            name=input_path.stem,
            ext=ext,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H%M%S"),
        )

    def convert(
        self, input_file: str, output: Optional[str] = None
    ) -> ConversionResult:
        """Convert a single file to markdown using the markitdown backend."""
        # Use markitdown for conversion
        result = self._converter.convert(input_file)
        
        output_file = self._generate_output_name(input_file, output)
        if output_file:
            with open(output_file, "w", encoding=self.config["conversion"]["encoding"]) as f:
                f.write(result.text_content)

        return ConversionResult(input_file, output_file, result.text_content)

    def convert_many(
        self, input_files: List[str], output_pattern: Optional[str] = None
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
            results.append(self.convert(input_file, output))
        return results