import dv_processing as dv
import numpy as np

EVENT_DTYPE = np.dtype([
    ('t', np.int64),  # Timestamp
    ('x', np.uint16), # X-Koordinate
    ('y', np.uint16), # Y-Koordinate
    ('p', np.bool_)   # Polarität
])
# Wenn du einen EventStore von der Kamera bekommst:
def dv_store_to_numpy(event_store: dv.EventStore):
    """
    Der effizienteste Weg überhaupt: Nutzt den C++ Speicher direkt.
    """
    # 1. Koordinaten und Zeitstempel extrahieren
    # dv-processing gibt diese oft schon als numpy-ähnliche Arrays zurück
    ts = event_store.timestamps()

    coordinates = event_store.coordinates()
    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    
    ps = event_store.polarities()
    
    # 2. In unseren strukturierten Array packen (falls du alles gebündelt brauchst)
    count = len(ts)
    events = np.empty(count, dtype=EVENT_DTYPE)
    
    events['t'] = ts
    events['x'] = xs
    events['y'] = ys
    events['p'] = ps
    
    return events