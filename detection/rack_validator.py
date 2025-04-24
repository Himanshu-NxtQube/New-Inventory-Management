import os
import cv2
import csv
from typing import Dict, Optional, Any, List


class RackValidator:
    def __init__(self, csv_path: str):
        """Initialize the RackValidator with a CSV file containing rack position data.

        Args:
            csv_path: Path to the CSV file containing rack position grid
        """
        self.csv_path = csv_path
        self.rack_positions: Dict[str, Dict[str, Any]] = {}
        self.grid: List[List[str]] = []
        self.load_rack_positions()


    def load_rack_positions(self) -> None:
        """Load rack positions from CSV file into memory.

        The CSV is treated as a grid where each rack's relationships are determined
        by its position relative to other cells in the grid.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Rack positions CSV file not found: {self.csv_path}")

        try:
            # First, load the entire CSV as a grid
            with open(self.csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                self.grid = [row for row in reader if any(cell.strip() for cell in row)]

            # Process the grid to build relationships
            for row_idx in range(len(self.grid)):
                for col_idx in range(len(self.grid[row_idx])):
                    current_rack = self.grid[row_idx][col_idx].strip()
                    if not current_rack or current_rack in ['NEW', 'Module1', 'Module2', 'Module3', 'Module4', 'Module5']:
                        continue

                    # Initialize relationships dictionary
                    self.rack_positions[current_rack] = {
                        'left': '',
                        'right': '',
                        'upper': '',
                        'lower': ''
                    }

                    # Find left adjacent (look for nearest non-empty cell to the left)
                    for left_idx in range(col_idx - 1, -1, -1):
                        if self.grid[row_idx][left_idx].strip() and \
                           self.grid[row_idx][left_idx] not in ['NEW', 'Module1', 'Module2', 'Module3', 'Module4', 'Module5']:
                            self.rack_positions[current_rack]['left'] = self.grid[row_idx][left_idx].strip()
                            break

                    # Find right adjacent (look for nearest non-empty cell to the right)
                    for right_idx in range(col_idx + 1, len(self.grid[row_idx])):
                        if self.grid[row_idx][right_idx].strip() and \
                           self.grid[row_idx][right_idx] not in ['NEW', 'Module1', 'Module2', 'Module3', 'Module4', 'Module5']:
                            self.rack_positions[current_rack]['right'] = self.grid[row_idx][right_idx].strip()
                            break

                    # Find upper rack (look for nearest non-empty cell above)
                    for upper_idx in range(row_idx - 1, -1, -1):
                        if upper_idx >= 0 and col_idx < len(self.grid[upper_idx]):
                            if self.grid[upper_idx][col_idx].strip() and \
                               self.grid[upper_idx][col_idx] not in ['NEW', 'Module1', 'Module2', 'Module3', 'Module4', 'Module5']:
                                self.rack_positions[current_rack]['upper'] = self.grid[upper_idx][col_idx].strip()
                                break

                    # Find lower rack (look for nearest non-empty cell below)
                    for lower_idx in range(row_idx + 1, len(self.grid)):
                        if lower_idx < len(self.grid) and col_idx < len(self.grid[lower_idx]):
                            if self.grid[lower_idx][col_idx].strip() and \
                               self.grid[lower_idx][col_idx] not in ['NEW', 'Module1', 'Module2', 'Module3', 'Module4', 'Module5']:
                                self.rack_positions[current_rack]['lower'] = self.grid[lower_idx][col_idx].strip()
                                break

        except Exception as e:
            raise ValueError(f"Error loading rack positions from CSV: {str(e)}")


    def validate_rack_id(self, rack_id: str) -> bool:
        """Validate if a rack ID exists in the system."""
        if not isinstance(rack_id, str):
            return False
        return rack_id in self.rack_positions


    def get_rack_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all rack positions and their relationships."""
        return self.rack_positions


    def get_left_adjacent_rack(self, rack_id: str) -> Optional[str]:
        """Get the rack ID of the left adjacent rack."""
        if not self.validate_rack_id(rack_id):
            return None
        return self.rack_positions[rack_id].get('left')


    def get_right_adjacent_rack(self, rack_id: str) -> Optional[str]:
        """Get the rack ID of the right adjacent rack."""
        if not self.validate_rack_id(rack_id):
            return None
        return self.rack_positions[rack_id].get('right')


    def get_upper_rack(self, rack_id: str) -> Optional[str]:
        """Get the rack ID of the upper rack."""
        if not self.validate_rack_id(rack_id):
            return None
        return self.rack_positions[rack_id].get('upper')


    def get_lower_rack(self, rack_id: str) -> Optional[str]:
        """Get the rack ID of the lower rack."""
        if not self.validate_rack_id(rack_id):
            return None
        return self.rack_positions[rack_id].get('lower')


    def validate_rack_relationship(self, rack1: str, rack2: str, relationship: str) -> bool:
        """Validate if two racks have the specified relationship."""
        if not self.validate_rack_id(rack1) or not self.validate_rack_id(rack2):
            return False

        if relationship not in ['left', 'right', 'upper', 'lower']:
            raise ValueError("Invalid relationship type. Must be 'left', 'right', 'upper', or 'lower'")

        return self.rack_positions[rack1].get(relationship) == rack2


    def get_rack_neighbors(self, rack_id: str) -> Dict[str, str]:
        """Get all neighboring racks for a given rack ID."""
        if not self.validate_rack_id(rack_id):
            return {}

        return self.rack_positions[rack_id]


    def validate_rack_pattern(self, q1: str, q2: str, q3: str, q4: str) -> bool:
        """Validate if four racks form a valid pattern (q1-q2 above q3-q4)."""
        if any(not self.validate_rack_id(q) for q in [q1, q2, q3, q4] if q != 'Unable to decode'):
            return False

        # Validate horizontal relationships
        if q1 != 'Unable to decode' and q2 != 'Unable to decode':
            if self.get_right_adjacent_rack(q1) != q2:
                return False

        if q3 != 'Unable to decode' and q4 != 'Unable to decode':
            if self.get_right_adjacent_rack(q3) != q4:
                return False

        # Validate vertical relationships
        if q1 != 'Unable to decode' and q3 != 'Unable to decode':
            if self.get_lower_rack(q1) != q3:
                return False

        if q2 != 'Unable to decode' and q4 != 'Unable to decode':
            if self.get_lower_rack(q2) != q4:
                return False

        return True