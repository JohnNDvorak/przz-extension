"""
Persistent leaderboard for tracking top kappa configurations.

Stores the top N configurations discovered during exploration sessions.
Data is persisted to a JSON file to survive across restarts.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import os


@dataclass
class LeaderboardEntry:
    """A single leaderboard entry."""
    kappa: float
    c: float
    R: float
    P1_tilde: List[float]
    P2_tilde: List[float]
    P3_tilde: List[float]
    Q_coeffs: Dict[str, float]  # Stored as {str(k): v} for JSON compatibility
    timestamp: str
    source: str = "manual"  # "manual", "heatmap", "optimizer", etc.
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "LeaderboardEntry":
        """Create from dictionary."""
        return cls(**d)


class Leaderboard:
    """
    Persistent leaderboard for top kappa configurations.

    Stores entries in a JSON file and maintains top N entries.
    """

    DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "leaderboard.json"
    MAX_ENTRIES = 20  # Keep top 20 entries

    def __init__(self, path: Optional[Path] = None):
        """
        Initialize leaderboard.

        Args:
            path: Path to JSON file. Defaults to data/leaderboard.json
        """
        self.path = Path(path) if path else self.DEFAULT_PATH
        self.entries: List[LeaderboardEntry] = []
        self._load()

    def _load(self):
        """Load entries from disk."""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                self.entries = [LeaderboardEntry.from_dict(e) for e in data.get("entries", [])]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load leaderboard: {e}")
                self.entries = []
        else:
            self.entries = []

    def _save(self):
        """Save entries to disk."""
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "entries": [e.to_dict() for e in self.entries]
        }

        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_entry(
        self,
        kappa: float,
        c: float,
        R: float,
        P1_tilde: List[float],
        P2_tilde: List[float],
        P3_tilde: List[float],
        Q_coeffs: Dict,
        source: str = "manual",
        notes: str = "",
    ) -> bool:
        """
        Add a new entry to the leaderboard.

        Args:
            kappa: Computed kappa value
            c: Computed c value
            R: R parameter used
            P1_tilde: P1 tilde coefficients
            P2_tilde: P2 tilde coefficients
            P3_tilde: P3 tilde coefficients
            Q_coeffs: Q coefficients
            source: How this configuration was discovered
            notes: Optional notes about this configuration

        Returns:
            True if entry was added (made it into top N), False otherwise
        """
        # Convert Q_coeffs keys to strings for JSON compatibility
        Q_coeffs_str = {str(k): v for k, v in Q_coeffs.items()}

        entry = LeaderboardEntry(
            kappa=kappa,
            c=c,
            R=R,
            P1_tilde=list(P1_tilde),
            P2_tilde=list(P2_tilde),
            P3_tilde=list(P3_tilde),
            Q_coeffs=Q_coeffs_str,
            timestamp=datetime.now().isoformat(),
            source=source,
            notes=notes,
        )

        # Check if this is a duplicate (same coefficients)
        for existing in self.entries:
            if (existing.P1_tilde == entry.P1_tilde and
                existing.P2_tilde == entry.P2_tilde and
                existing.P3_tilde == entry.P3_tilde and
                abs(existing.R - entry.R) < 1e-6):
                # Duplicate - update if better kappa
                if entry.kappa > existing.kappa:
                    existing.kappa = entry.kappa
                    existing.c = entry.c
                    existing.timestamp = entry.timestamp
                    existing.source = entry.source
                    existing.notes = entry.notes
                    self._sort_and_trim()
                    self._save()
                    return True
                return False

        # Add new entry
        self.entries.append(entry)
        self._sort_and_trim()
        self._save()

        # Check if entry made it into top N
        return entry in self.entries

    def _sort_and_trim(self):
        """Sort by kappa descending and keep only top N entries."""
        self.entries.sort(key=lambda e: e.kappa, reverse=True)
        self.entries = self.entries[:self.MAX_ENTRIES]

    def get_top(self, n: int = 10) -> List[LeaderboardEntry]:
        """Get top N entries."""
        return self.entries[:n]

    def get_best(self) -> Optional[LeaderboardEntry]:
        """Get the best entry."""
        return self.entries[0] if self.entries else None

    def clear(self):
        """Clear all entries."""
        self.entries = []
        self._save()

    def remove_entry(self, index: int) -> bool:
        """Remove entry at index."""
        if 0 <= index < len(self.entries):
            del self.entries[index]
            self._save()
            return True
        return False

    def get_przz_rank(self, przz_kappa: float = 0.417293962) -> Optional[int]:
        """Get where PRZZ baseline would rank."""
        for i, entry in enumerate(self.entries):
            if entry.kappa < przz_kappa:
                return i + 1  # 1-indexed rank
        return len(self.entries) + 1 if self.entries else 1


# Global instance for easy access
_leaderboard: Optional[Leaderboard] = None


def get_leaderboard() -> Leaderboard:
    """Get the global leaderboard instance."""
    global _leaderboard
    if _leaderboard is None:
        _leaderboard = Leaderboard()
    return _leaderboard
