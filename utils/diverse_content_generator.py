import random
import string
from typing import Callable, Optional


class DiverseContentGenerator:
    """
    Generates diverse content for context testing (ABC Strategy).
    - Version A: Structured / Technical (Code, Logs, JSON)
    - Version B: Narrative / Literary (Stories, News, Conversations)
    - Version C: Data-dense / Informational (Specs, Tables, Documentation)
    """

    @staticmethod
    def get_version_a(token_count: int) -> str:
        """Technical / Code representation"""
        patterns = [
            "def process_data(item):\n    return item * 2\n",
            "class DataNode:\n    def __init__(self, val):\n        self.val = val\n",
            '{{ "status": "success", "data": {{ "id": 123, "meta": "test" }} }}\n',
            "[2026-01-08 15:36:55] INFO: Worker started. ID={}\n",
        ]
        content = ""
        while len(content.split()) < token_count: # Simple word-based approx for initialization
            content += patterns[random.randint(0, len(patterns)-1)].format(random.randint(100, 999))
        return content

    @staticmethod
    def get_version_b(token_count: int) -> str:
        """Narrative / Literary representation"""
        phrases = [
            "The quick brown fox jumps over the lazy dog. ",
            "Once upon a time, in a land far away, there lived a wise old owl. ",
            "The rain in Spain stays mainly in the plain. ",
            "In the middle of the night, the stars shine bright above the mountains. ",
        ]
        content = ""
        while len(content.split()) < token_count:
            content += phrases[random.randint(0, len(phrases)-1)]
        return content

    @staticmethod
    def get_version_c(token_count: int) -> str:
        """Data-dense / Documentation representation"""
        lines = [
            "SPECIFICATION 1.0: Hardware requirements must meet Tier 3 standards. ",
            "TABLE ROW {}: ID_CODE_99 | STATUS_ACTIVE | PRIORITY_HIGH | LATENCY_LOW. ",
            "DOCUMENTATION: Ensure all endpoints are authenticated using Bearer tokens. ",
            "METRICS: CPU_USAGE=45%, MEM_FREE=12GB, DISK_IO=230MB/s. ",
        ]
        content = ""
        while len(content.split()) < token_count:
            content += lines[random.randint(0, len(lines)-1)].format(random.randint(1000, 9999))
        return content

    @classmethod
    def generate(cls, version: str, token_count: int) -> str:
        if version.upper() == 'A':
            return cls.get_version_a(token_count)
        elif version.upper() == 'B':
            return cls.get_version_b(token_count)
        elif version.upper() == 'C':
            return cls.get_version_c(token_count)
        return cls.get_version_b(token_count)

    @classmethod
    def generate_calibrated(
        cls,
        version: str,
        target_tokens: int,
        token_counter: Callable[[str], int],
        max_iterations: int = 50
    ) -> str:
        """
        Generate content with precise token calibration.
        
        Args:
            version: Content version ('A', 'B', or 'C')
            target_tokens: Exact number of tokens to generate
            token_counter: Function that counts tokens in a string
            max_iterations: Maximum calibration iterations
            
        Returns:
            Content string with exactly target_tokens tokens (or as close as possible)
        """
        if target_tokens <= 0:
            return ""
        
        # Initial rough generation (overshoot by 30% to ensure enough content)
        rough_content = cls.generate(version, int(target_tokens * 1.3) + 10)
        
        current = rough_content
        chars_pool = string.ascii_letters + string.digits + " .,;:!?"
        
        for _ in range(max_iterations):
            count = token_counter(current)
            diff = count - target_tokens
            
            if diff == 0:
                return current
            elif diff > 0:
                # Too long: trim from end (character by character for precision)
                # Estimate ~3-4 chars per token for English
                trim_chars = max(1, int(abs(diff) * 2))
                if len(current) > trim_chars:
                    current = current[:-trim_chars]
                else:
                    # Content too short to trim, regenerate
                    current = cls.generate(version, int(target_tokens * 1.5))
            else:
                # Too short: add characters
                add_chars = max(1, int(abs(diff) * 3))
                # Add content-appropriate padding
                if version.upper() == 'A':
                    padding = f"\n# Comment block {random.randint(100, 999)}\n"
                elif version.upper() == 'C':
                    padding = f" DATA_FIELD_{random.randint(1000, 9999)}=VALUE "
                else:
                    padding = " " + "".join(random.choices(chars_pool, k=add_chars))
                current = current + padding
        
        # Final adjustment: if still not exact, do fine-grained char adjustment
        final_count = token_counter(current)
        while final_count > target_tokens and len(current) > 1:
            current = current[:-1]
            final_count = token_counter(current)
        
        return current

