import time
from typing import Dict, List, Optional

from ..utils.config import settings
from ..utils.logger import logger


class TranslateError(Exception):
    """Raised when translation fails"""
    pass


class Translator:
    def __init__(self, config=None):
        self.config = config or settings.translation
        self.translator = None
        
    def load_translator(self):
        """Load the translation model"""
        if not self.config.enabled:
            return
            
        if self.translator is not None:
            return
            
        logger.info(f"Loading {self.config.engine} translator")
        
        if self.config.engine == "whisper":
            # Whisper handles translation during transcription
            pass
        elif self.config.engine == "argos":
            self._load_argos_translator()
        else:
            raise ValueError(f"Unknown translation engine: {self.config.engine}")
            
    def _load_argos_translator(self):
        """Load Argos Translate"""
        try:
            import argostranslate.package
            import argostranslate.translate
            
            # Download and install translation packages if needed
            from_code = "auto"  # Will be determined from source
            to_code = self.config.target_language
            
            # For now, we'll load on-demand when we know the source language
            self.argos_translate = argostranslate.translate
            
        except ImportError as e:
            raise TranslateError("Argos Translate not installed") from e
            
    def translate(self, text: str, source_language: Optional[str] = None) -> Dict:
        """Translate text to target language"""
        if not self.config.enabled:
            logger.info("Translation is disabled")
            return {
                "text": text,
                "source_language": source_language,
                "target_language": source_language,
                "translated": False,
            }
            
        if source_language == self.config.target_language:
            logger.info(f"Source and target language are the same: {source_language}")
            return {
                "text": text,
                "source_language": source_language,
                "target_language": source_language,
                "translated": False,
            }
            
        self.load_translator()
        
        logger.info(f"Translating from {source_language} to {self.config.target_language}")
        start_time = time.time()
        
        try:
            if self.config.engine == "whisper":
                # Whisper translation happens during transcription
                result = {
                    "text": text,
                    "source_language": source_language,
                    "target_language": self.config.target_language,
                    "translated": True,
                    "engine": "whisper",
                }
            else:
                result = self._translate_argos(text, source_language)
                
            duration = time.time() - start_time
            result["duration"] = duration
            logger.info(f"Translation completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslateError(f"Translation failed: {e}") from e
            
    def _translate_argos(self, text: str, source_language: str) -> Dict:
        """Translate using Argos Translate"""
        import argostranslate.package
        import argostranslate.translate
        
        # Map language codes if needed
        source_code = self._map_language_code(source_language)
        target_code = self._map_language_code(self.config.target_language)
        
        # Get available packages
        available_packages = argostranslate.package.get_available_packages()
        
        # Find the translation package
        package_to_install = None
        for package in available_packages:
            if package.from_code == source_code and package.to_code == target_code:
                package_to_install = package
                break
                
        if not package_to_install:
            raise TranslateError(
                f"No translation package available for {source_code} -> {target_code}"
            )
            
        # Download and install package if not already installed
        installed_packages = argostranslate.package.get_installed_packages()
        if package_to_install not in installed_packages:
            logger.info(f"Downloading translation package: {source_code} -> {target_code}")
            argostranslate.package.install_from_path(package_to_install.download())
            
        # Get translation function
        translation = argostranslate.translate.get_translation_from_codes(
            source_code, target_code
        )
        
        if not translation:
            raise TranslateError(f"Failed to load translator for {source_code} -> {target_code}")
            
        # Translate text
        translated_text = translation.translate(text)
        
        return {
            "text": translated_text,
            "source_language": source_language,
            "target_language": self.config.target_language,
            "translated": True,
            "engine": "argos",
        }
        
    def _map_language_code(self, code: str) -> str:
        """Map language codes to Argos format"""
        # Common mappings
        mappings = {
            "zh": "zh",
            "zh-cn": "zh",
            "zh-tw": "zh",
            "en": "en",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "ja": "ja",
            "ko": "ko",
            "pt": "pt",
            "ru": "ru",
            "ar": "ar",
            "hi": "hi",
        }
        
        return mappings.get(code.lower(), code.lower())
        
    def translate_segments(self, segments: List[Dict], source_language: str) -> List[Dict]:
        """Translate transcript segments"""
        if not self.config.enabled or source_language == self.config.target_language:
            return segments
            
        translated_segments = []
        
        for segment in segments:
            try:
                result = self.translate(segment["text"], source_language)
                translated_segment = segment.copy()
                translated_segment["text"] = result["text"]
                translated_segment["original_text"] = segment["text"]
                translated_segments.append(translated_segment)
            except Exception as e:
                logger.warning(f"Failed to translate segment: {e}")
                translated_segments.append(segment)
                
        return translated_segments