from tqdm import tqdm
from event_types import dv_store_to_numpy
import numpy as np
from pathlib import Path
from typing import List
import dv_processing as dv

def get_aedat4_files(folder: Path) -> List[Path]:
    """Reads a folder and returns a list of all .aedat4 files."""
    aedat4_files = [f for f in folder.iterdir() if f.suffix == '.aedat4']
    return aedat4_files

def proccess_aedat4_file(file_path: Path) -> np.ndarray:
    """Processes an AEDAT4 file and extracts event data as a numpy array."""
    reader = dv.io.MonoCameraRecording(str(file_path))
    numpy_batches = [] 
    total_events_count = 0
    while reader.isRunning():
        events_store = reader.getNextEventBatch()
        
        if events_store is not None and events_store.size() > 0:
            events_arr = dv_store_to_numpy(events_store)
            
            numpy_batches.append(events_arr)
            total_events_count += len(events_arr)
    
    return np.concatenate(numpy_batches) if numpy_batches else np.array([])
            
def main():
    # Define paths
    save_path = Path('data/binary')
    raw_path  = Path('data/raw')

    # Get list of AEDAT4 files
    aedat4_files = get_aedat4_files(raw_path)
    
    print("AEDAT4 Files: ", aedat4_files)
    
    for source in tqdm(aedat4_files, desc="Processing AEDAT4 files"):
        filename = source.stem

        if (save_path / f"{filename}").exists():
            continue

        # Process the AEDAT4 file
        event_data = proccess_aedat4_file(source)

        np.save(save_path / f"{filename}", event_data, allow_pickle=True)


if __name__ == "__main__":
    main()