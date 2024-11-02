from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import wave
import soundfile as sf
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


class AudioLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_with_wave(self):
        """Load audio using wave library (basic approach)"""
        with wave.open(self.file_path, 'rb') as wav_file:
            # Get basic properties
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read raw audio data
            raw_data = wav_file.readframes(n_frames)
            
            # Convert raw data to numpy array
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            else:
                raise ValueError("Unsupported sample width")
                
            audio_data = np.frombuffer(raw_data, dtype=dtype)
            
            # Reshape if stereo
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                
        return audio_data, frame_rate


class ClickDetector:
    def __init__(self, audio_loader):
        self.audio_loader = audio_loader
        self.clicks = []
        self.csv_annotations = []
    def simple_click_detector(self):
        """Simple single-click detection"""
        fig, ax = plt.subplots()
        ax.set_title('Click anywhere on the plot (press "q" to quit)')
        
        # Plot some sample data if needed
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        
        def onclick(event):
            if event.button == 3:  # Left click
                if event.inaxes is not None:  # Click is within the axes
                    self.clicks.append((event.xdata, event.ydata))
                    print(f'Clicked at x={event.xdata:.2f}, y={event.ydata:.2f}')
                    
                    # Optional: Plot a marker at click position
                    ax.plot(event.xdata, event.ydata, 'ro')
                    fig.canvas.draw()
        
        def onkey(event):
            if event.key == 'q':
                plt.close()
        
        # Connect the event handlers
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show()
        
    def interactive_annotator(self):
        """More advanced click detector with annotations"""
        audio_data, rate = self.audio_loader.load_with_wave()
        fig, ax = plt.subplots()
        ax.set_title('Click to add points. Right click to remove.\nPress "q" to quit')
        
        points, = ax.plot(np.linspace(0, len(audio_data)/rate, len(audio_data)), audio_data, 'rx-', markersize=0.5, linewidth=0.5)  # Empty line with red dots
        annotations = []
        def onclick(event):
            if event.inaxes is not None:
                if event.button == 3:  # Right click
                    # Add point
                    
                    plt.plot(event.xdata, event.ydata, 'go', picker=5)
                    
                    # Add annotation
                    annot = ax.annotate(
                        f'({event.xdata:.2f}, {event.ydata:.2f})',
                        (event.xdata, event.ydata),
                        xytext=(5, 5), textcoords='offset points'
                    )
                    annotations.append(annot)
                    self.csv_annotations.append((event.xdata, event.ydata))
                    print(f'Clicked at x={event.xdata:.2f}, y={event.ydata:.2f}')
                
                fig.canvas.draw()
        
        def onkey(event):
            if event.key == 'q':
                plt.close()
        
        # Connect the event handlers
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show()

def main():    
    file_path = "2024-11-02_13-06-20_audio_record.wav"
    loader = AudioLoader(file_path)
    samples, rate = loader.load_with_wave()
    print(f"Wave: Sample rate: {rate}Hz, Shape: {samples.shape}")
    
    detector = ClickDetector(loader)
    
    # Choose one of these methods:
    detector.interactive_annotator()  # Simple click detection
    # detector.interactive_annotator()  # Advanced with annotations
    
    print("Final click positions:", detector.clicks)
    path = file_path.replace(".wav", ".csv")
    dataframe = pd.DataFrame(detector.csv_annotations, columns=["time", "amplitude"])
    dataframe.to_csv(path, index=False)

if __name__ == "__main__":
    main()