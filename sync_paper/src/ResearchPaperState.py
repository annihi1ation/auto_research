from dataclasses import dataclass


@dataclass
class ResearchPaperState:
    dataset_name: str = "Cornell-University/arxiv"
    save_path: str = "data"
