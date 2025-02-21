import gzip
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from rl4llm.core.data_types import SampleLog


class FileHandler:
    """Simple file handler that manages a single file with buffered writing"""

    def __init__(self, save_path: str, file_type: str = 'csv', compress: bool = True):
        if file_type not in ['csv', 'jsonl']:
            raise ValueError("file_type must be either 'csv' or 'jsonl'")

        self.compress = compress
        # Add .gz extension if compression is enabled
        self.out_file = f"{save_path}.gz" if compress else save_path
        self.file_type = file_type
        self.buffer = []
        self._is_header_written_csv = False if file_type == 'csv' else True  # track csv header

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.out_file), exist_ok=True)

    def log_entry(self, data: Dict):
        """Add an entry to the buffer"""
        self.buffer.append(data)

    def flush(self):
        """Write buffered data to file"""
        if not self.buffer:
            return

        if self.file_type == 'csv':
            df = pd.DataFrame(self.buffer)
            write_header = not self._is_header_written_csv

            if self.compress:
                df.to_csv(
                    self.out_file, mode='a' if not write_header else 'w', header=write_header, index=False, compression='gzip'
                )
            else:
                df.to_csv(
                    self.out_file,
                    mode='a' if not write_header else 'w',
                    header=write_header,
                    index=False,
                )
            if write_header:
                self._is_header_written_csv = True

        else:  # jsonl
            mode = 'at' if self.compress else 'a'
            opener = gzip.open if self.compress else open

            with opener(self.out_file, mode, encoding='utf-8') as f:
                for item in self.buffer:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

        self.buffer.clear()

    def close(self):
        """Flush remaining data and close"""
        self.flush()
