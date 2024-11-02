import dataclasses
import multiprocessing as mp


class VideoQueueItem:
    device_id: str
    video_path: str


class Worker(mp.Process):
    def __init__(self, processing_fn, queue: mp.Queue):
        super().__init__()
        self.processing_fn = processing_fn
        self.input_queue = mp.Queue(maxsize=4)
        self.output_queue = mp.Queue(maxsize=4)
        self.stop_event = mp.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get()
                output = self.processing_fn(item)
                self.output_queue.put(output)
            except Exception as e:
                print(e)

    def stop(self):
        self.stop_event.set()

    def put(self, item):
        self.input_queue.put(item)

    def get(self):
        return self.output_queue.get()
