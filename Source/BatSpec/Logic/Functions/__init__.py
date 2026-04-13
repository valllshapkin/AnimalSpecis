from BatSpec.ArrType import Array, Float32
from typing import Any, Callable
import numpy as np
from BatSpec.Logic.SaveIntegral import SaveIntegralProtocol

type T = Any
type H = Any

type TimeFuncShower = Callable[[TimeFunc[Any]], None]
type SpecFuncShower = Callable[[SpecFunc[Any]], None]

GlobalShowTimeFunc: TimeFuncShower
GlobalShowSpecFunc: SpecFuncShower

class TimeFunc[Float](SaveIntegralProtocol):
    FW = np

    def __init__(self, data: Array[Float, T] , sr: int):
        self.data = data
        self.sr = sr
        
    def SaveIntegralEnergy(self) -> float:
        return self.FW.sum(self.data ** 2) / self.sr

    def SaveIntegralArea(self) -> float:
        return self.FW.sum(self.data) / self.sr

    def withFrameWork(self, fw):
        self.FW = fw
        return self

    @property
    def duration(self) -> float:
        return len(self.data) / self.sr

    @property
    def time_axis(self):
        return np.arange(len(self.data), dtype=np.float64) / self.sr
    
class SpecFunc[Float](SaveIntegralProtocol):
    FW = np

    def __init__(self, 
        matrix: Array[Float, T, H], 
        freq: Array[Float, H], 
        time: Array[Float, T]
    ):
        self.matrix = matrix
        self.freq = freq
        self.time = time

    def __mul__(self, other):
        if self.freq.shape != other.freq.shape or not np.all(self.freq == other.freq):
            raise RuntimeError("Frequencies or times do not match")
        return SpecFunc(self.matrix * other.matrix, self.freq, other.time)
    
    def __add__(self, other):
        if self.freq.shape != other.freq.shape or not np.all(self.freq == other.freq):
            raise RuntimeError("Frequencies or times do not match")
        return SpecFunc(self.matrix + other.matrix, self.freq, other.time)

    @property
    def dt(self) -> float:
        return self.time[1] - self.time[0]
    
    @property
    def df(self) -> float:
        return self.freq[1] - self.freq[0]

    def SaveIntegralEnergy(self):
        return self.FW.sum(self.matrix**2) * self.dt * self.df
    
    def SaveIntegralArea(self):
        return self.FW.sum(self.matrix) * self.dt * self.df
    
    def cloneApply[New](self, func: Callable[[Array[Float, T]], Array[New, T]]) -> 'SpecFunc[New]':
        return SpecFunc(func(self.matrix), self.freq, self.time)
    
    def showLog(self):
        GlobalShowSpecFunc(self.cloneApply(
            lambda arr: self.FW.log(arr)
        ))
    
    def show(self):
        GlobalShowSpecFunc(self)

    def save(self, filepath: str, compressed: bool = False):
        """
        Save the SpecFunc data to an npz file.
        
        Parameters:
        -----------
        filepath : str
            Path where to save the file (without extension)
        compressed : bool
            If True, use savez_compressed; if False, use savez
        """
        save_func = self.FW.savez_compressed if compressed else self.FW.savez
        
        save_func(
            filepath,
            matrix=self.matrix,
            freq=self.freq,
            time=self.time
        )

    @classmethod
    def load(cls, filepath: str):
        """
        Load a SpecFunc from an npz file.
        
        Parameters:
        -----------
        filepath : str
            Path to the npz file (with or without .npz extension)
        
        Returns:
        --------
        SpecFunc
            Loaded SpecFunc instance
        """
        data = cls.FW.load(filepath)
        return cls(
            matrix=data['matrix'],
            freq=data['freq'],
            time=data['time']
        )

    def savez(self, filepath: str):
        """Convenience method for standard save"""
        self.save(filepath, compressed=False)

    def savez_compressed(self, filepath: str):
        """Convenience method for compressed save"""
        self.save(filepath, compressed=True)

    def saveDebugLogPng(self, filepath: str, cmap: str = 'viridis', dpi: int = 150):
        """
        Save debug PNG image of the spectrogram in logarithmic scale.
        
        Parameters:
        -----------
        filepath : str
            Path where to save the PNG file (should end with .png)
        cmap : str
            Colormap name (default: 'viridis')
        dpi : int
            Resolution of the output image (default: 150)
        """
        import matplotlib.pyplot as plt
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
        
        # Apply logarithm to the matrix (handle zeros and negative values)
        log_matrix = self.FW.log(self.matrix + 1e-10)  # Small epsilon to avoid log(0)
        
        # Create extent for imshow [freq_min, freq_max, time_min, time_max]
        # Note: imshow expects extent as [left, right, bottom, top]
        # Typically: time on x-axis, frequency on y-axis
        extent = [
            self.time[0], self.time[-1],  # x-axis: time
            self.freq[0], self.freq[-1]   # y-axis: frequency
        ]
        
        # Plot the spectrogram
        im = ax.imshow(
            log_matrix.T,  # Transpose if needed (assuming matrix shape is [freq, time])
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=cmap
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Log Power', fontsize=12)
        
        # Labels and title
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Spectrogram (Log Scale) - Debug', fontsize=14, fontweight='bold')
        
        # Add grid for better readability (optional)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Save with tight layout
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Debug PNG saved to: {filepath}")

    def saveDebugLogPngWithStats(self, filepath: str, cmap: str = 'viridis', dpi: int = 150):
        """
        Save debug PNG with additional statistical information.
        
        Parameters:
        -----------
        filepath : str
            Path where to save the PNG file
        cmap : str
            Colormap name
        dpi : int
            Resolution of the output image
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogFormatter
        
        # Create figure with subplots for additional info
        fig = plt.figure(figsize=(14, 10), dpi=dpi)
        
        # Main spectrogram
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        # Apply logarithm with safe handling
        log_matrix = self.FW.log(self.matrix + 1e-10)
        
        extent = [self.time[0], self.time[-1], self.freq[0], self.freq[-1]]
        
        im = ax1.imshow(
            log_matrix.T,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=cmap
        )
        
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Spectrogram (Log Scale)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.2, linestyle='--')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Log Power', fontsize=10)
        
        # Statistics subplot (top-right)
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        ax2.axis('off')
        
        # Calculate statistics
        data_flat = self.matrix.flatten()
        log_data_flat = log_matrix.flatten()
        
        stats_text = f"""
        Spectrogram Statistics:
        -----------------------
        Shape: {self.matrix.shape}
        Time points: {len(self.time)}
        Freq points: {len(self.freq)}
        
        dt: {self.dt:.4e}
        df: {self.df:.4e}
        
        Original Data:
        Min: {data_flat.min():.4e}
        Max: {data_flat.max():.4e}
        Mean: {data_flat.mean():.4e}
        Std: {data_flat.std():.4e}
        
        Log Data:
        Min: {log_data_flat.min():.2f}
        Max: {log_data_flat.max():.2f}
        Mean: {log_data_flat.mean():.2f}
        
        Energy Integral: {self.SaveIntegralEnergy():.4e}
        Area Integral: {self.SaveIntegralArea():.4e}
        """
        
        ax2.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Histogram subplot (bottom)
        ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        
        # Plot histogram of log values
        ax3.hist(log_data_flat, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Log Power', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Distribution of Log Values', fontsize=12)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Add mean and median lines
        mean_log = log_data_flat.mean()
        median_log = np.median(log_data_flat)
        ax3.axvline(mean_log, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_log:.2f}')
        ax3.axvline(median_log, color='green', linestyle='--', linewidth=2, label=f'Median: {median_log:.2f}')
        ax3.legend()
        
        plt.suptitle(f'Debug Spectrogram - Log Scale\n{filepath}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Debug PNG with statistics saved to: {filepath}")