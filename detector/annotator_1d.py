from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import wave
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


class BadQuality:
    start: float | None = None
    end: float | None = None
    
    def add_point(self, x, y):
        if self.start is None:
            self.start = (x, y)
        else:
            self.end = (x, y)

    def get_segment(self):
        if self.start is None or self.end is None:
            return None
        return (self.start[0], self.end[0])

class ClickDetector:
    def __init__(self, audio_data, rate):
        self.audio_data = audio_data
        self.rate = rate
        self.clicks = []
        self.csv_annotations = []
        self.quality: bool = True
        self.bad_quality = BadQuality()
        self.bad_quality_segments = []
        
    def simple_click_detector(self):
        """Simple single-click detection"""
        fig, ax = plt.subplots(dpi=200)
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
        fig, ax = plt.subplots(dpi=200)
        ax.set_title('Click to add points. Right click to remove.\nPress "q" to quit')
        
        points, = ax.plot(np.linspace(0, len(self.audio_data)/self.rate, len(self.audio_data)), self.audio_data, 'rx-', markersize=0.5, linewidth=0.5)  # Empty line with red dots
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
                    self.csv_annotations.append((event.xdata, event.ydata, int(self.quality)))
                    print(f'Clicked at x={event.xdata:.2f}, y={event.ydata:.2f}')
                
                fig.canvas.draw()
        
        def onkey(event):
            if event.key == 'q':
                plt.close()
                
            if event.key == "b":
                mouse_x, mouse_y = event.inaxes.transData.inverted().transform((event.x, event.y))
                plt.plot(mouse_x, mouse_y, 'bo', picker=5)
                self.bad_quality.add_point(mouse_x, mouse_y)
                if self.bad_quality.get_segment() is not None:
                    self.bad_quality_segments.append(self.bad_quality.get_segment())
                    plt.plot((self.bad_quality.start[0], self.bad_quality.end[0]), (self.bad_quality.start[1], self.bad_quality.end[1]), 'b-', picker=5)
                    self.bad_quality = BadQuality()
                print(f"Mouse position: x={mouse_x:.2f}, y={mouse_y:.2f}")
                self.quality = False
            if event.key == "g":
                self.quality = True
            fig.canvas.draw()
        
        # Connect the event handlers
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show()

def main():    
    import json
    from pathlib import Path

    files = Path("/Users/brani/code/ACGTeam_HeartTrack_data/lavicka_data").glob("*.csv")
    for file_path in files:
        if file_path.name.endswith(".wav"):
            loader = AudioLoader(file_path)
            samples, rate = loader.load_with_wave()
            print(f"Wave: Sample rate: {rate}Hz, Shape: {samples.shape}")
            
            detector = ClickDetector(samples, rate)
        else:
            samples = np.loadtxt(file_path)
            detector = ClickDetector(samples, rate=30)
        # Choose one of these methods:
        detector.interactive_annotator()  # Simple click detection
        # detector.interactive_annotator()  # Advanced with annotations
        
        print("Final click positions:", detector.clicks)
        path = str(file_path).replace(".csv", ".json")
        points = [{"time": t} for t, a, q in detector.csv_annotations]
        bad_segments = [{"start": s, "end": e} for s, e in detector.bad_quality_segments]
        with open(path, "w") as f:
            json.dump({"points": points, "bad_segments": bad_segments}, f, indent=4)

if __name__ == "__main__":
    main()