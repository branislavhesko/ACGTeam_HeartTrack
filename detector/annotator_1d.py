from matplotlib import pyplot as plt
import numpy as np
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


file_path = "/home/brani/heart_data/HONORJSN-L21/2024-11-02_13-06-20_audio_record.wav"
loader = AudioLoader(file_path)
samples, rate = loader.load_with_wave()
print(f"Wave: Sample rate: {rate}Hz, Shape: {samples.shape}")


class ClickDetector:
    def __init__(self):
        self.clicks = []
        
    def simple_click_detector(self):
        """Simple single-click detection"""
        fig, ax = plt.subplots()
        ax.set_title('Click anywhere on the plot (press "q" to quit)')
        
        # Plot some sample data if needed
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        
        def onclick(event):
            if event.button == 2:  # Left click
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
        fig, ax = plt.subplots()
        ax.set_title('Click to add points. Right click to remove.\nPress "q" to quit')
        
        points, = ax.plot([], [], 'ro', picker=5)  # Empty line with red dots
        annotations = []
        
        def onclick(event):
            if event.inaxes is not None:
                if event.button == 1:  # Left click
                    # Add point
                    xdata = list(points.get_xdata())
                    ydata = list(points.get_ydata())
                    xdata.append(event.xdata)
                    ydata.append(event.ydata)
                    points.set_data(xdata, ydata)
                    
                    # Add annotation
                    annot = ax.annotate(
                        f'({event.xdata:.2f}, {event.ydata:.2f})',
                        (event.xdata, event.ydata),
                        xytext=(5, 5), textcoords='offset points'
                    )
                    annotations.append(annot)
                    
                elif event.button == 3:  # Right click
                    # Remove closest point
                    xdata = list(points.get_xdata())
                    ydata = list(points.get_ydata())
                    if len(xdata) > 0:
                        # Find closest point
                        distances = [(x - event.xdata)**2 + (y - event.ydata)**2 
                                   for x, y in zip(xdata, ydata)]
                        idx = distances.index(min(distances))
                        
                        # Remove point and annotation
                        xdata.pop(idx)
                        ydata.pop(idx)
                        points.set_data(xdata, ydata)
                        
                        if annotations:
                            annotations[idx].remove()
                            annotations.pop(idx)
                
                fig.canvas.draw()
        
        def onkey(event):
            if event.key == 'q':
                plt.close()
        
        # Connect the event handlers
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show()

def main():
    detector = ClickDetector()
    
    # Choose one of these methods:
    detector.simple_click_detector()  # Simple click detection
    # detector.interactive_annotator()  # Advanced with annotations
    
    print("Final click positions:", detector.clicks)


if __name__ == "__main__":
    main()