from typing import Dict, List
import json
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


class DPODataset(Dataset):
    def __init__(self, file_path: str, max_samples: int | None = None):
        self.examples: List[DPOExample] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                self.examples.append(
                    DPOExample(
                        prompt=obj["prompt"],
                        chosen=obj["chosen"],
                        rejected=obj["rejected"],
                    )
                )
                if max_samples is not None and len(self.examples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        return {
            "prompt": ex.prompt,
            "chosen": ex.chosen,
            "rejected": ex.rejected,
        }
