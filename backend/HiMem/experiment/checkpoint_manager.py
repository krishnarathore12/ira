import os
import pickle

SAVE_INTERVAL = 10


class ProcessingState:
    """Holds the current progress of the data processing job."""

    def __init__(self, all_units):
        self.all_units = all_units
        self.finished_units = set()

    def get_finished_units(self):
        return self.finished_units

    def is_processed(self, key):
        if key not in self.finished_units:
            return False
        return True

    def record(self, unit):
        """Records a successful unit and its result."""
        self.finished_units.add(unit)

    def is_complete(self):
        """Checks if all units have been processed."""
        return len(self.finished_units) == len(self.all_units)


class ProcessingStateManager:
    def __init__(self, name):
        self.name = name
        self.checkpoint_file = f"{self.name}_processing_checkpoint.pkl"

    def save_checkpoint(self, state):
        """Saves the current state object to a pickle file."""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(state, f)
            print(
                f"\nCheckpoint saved successfully to {self.checkpoint_file}. Finished units: {len(state.finished_units)}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, all_units):
        """Loads the state from a pickle file if it exists, otherwise initializes a new state."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    state = pickle.load(f)
                # Optional check: ensure the loaded state matches the current data set
                if set(state.all_units) != set(all_units):
                    print("Warning: Data set changed. Starting fresh.")
                    return ProcessingState(all_units)
                print(
                    f"⏳ Loaded checkpoint from {self.checkpoint_file}. Resuming from unit {len(state.finished_units)}.")
                return state
            except Exception as e:
                print(f"Error loading checkpoint ({e}). Starting fresh.")
                return ProcessingState(all_units)
        else:
            print("No checkpoint found. Starting new processing job.")
            return ProcessingState(all_units)
