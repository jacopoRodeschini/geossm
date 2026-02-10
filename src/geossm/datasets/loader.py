from importlib import resources
import pandas as pd

_AVAILABLE = {
    "agrimonia": "agrimonia.csv"
}

def load_dataset(name: str):
    if name not in _AVAILABLE:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {list(_AVAILABLE)}"
        )

    filename = _AVAILABLE[name]

    with resources.files("geossm.datasets.data").joinpath(filename).open("rb") as f:
        if filename.endswith(".csv"):
            return pd.read_csv(f)
        else:
            raise RuntimeError("Unsupported format")
        
def list_datasets(): 
    """
    List available example datasets bundled with the package.

    Returns
    -------
    str
        Formatted table with dataset name, file extension and size in MB.
    """
    
    base = resources.files("geossm.datasets.data")

    header = f"{'Dataset':<20} {'Ext':<6} {'Size (MB)':>10}"
    sep = f"{'-' * len(header)}"

    rows = [header, sep]

    # Add a flag if the dataset is larger than 10 MB
    
    for name in sorted(_AVAILABLE):
        filename = _AVAILABLE[name]
        path = base / filename
        size_mb = path.stat().st_size / 1024**2
        ext = path.suffix.lstrip(".")

        flag = " *" if size_mb > 10 else ""
        rows.append(
            f"{name:<20} {ext:<6} {size_mb:>10.3f}{flag}"
        )

    return "\n".join(rows)