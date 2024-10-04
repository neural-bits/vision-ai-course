import time
import av
import cv2
import numpy as np
from core.dtypes import FrameBuffer
from utils.logger import get_logger

logger = get_logger("FrameAcquisition")


class FrameAcquisition:
    def __init__(self, shared_buffer: FrameBuffer, video_path: str):
        self.shared_buffer = shared_buffer
        self.video_path = video_path
        logger.info(f"Attached to shared memory {self.shared_buffer.shm.name}")

    def acquire_frames(self):
        try:
            capture = av.open(self.video_path)
            logger.info(f"Opened video file {self.video_path}")
            
            frame_count = 0
            start_time = time.time()  # Start the timer

            for packet in capture.demux():
                for frame in packet.decode():
                    if isinstance(frame, av.VideoFrame):
                        self.shared_buffer.space_available.acquire()

                        next_index = (
                            self.shared_buffer.latest_index.value + 1
                        ) % self.shared_buffer.buffer_size
                        image = frame.to_rgb().to_ndarray()

                        np.copyto(self.shared_buffer.buffer[next_index], image)
                        self.shared_buffer.index_buffer[next_index] = next_index
                        self.shared_buffer.latest_index.value = next_index

                        logger.info(f"Produced frame {next_index}")
                        self.shared_buffer.frame_available.release()

                        # FPS calculation
                        frame_count += 1
                        elapsed_time = time.time() - start_time

                        fps = frame_count / elapsed_time
                        logger.info(f"Current FPS: {fps:.2f}")
                        frame_count = 0
                        start_time = time.time()

            # Optional: if you want to print the final FPS after finishing
            if frame_count > 0:
                final_fps = frame_count / (time.time() - start_time)
                logger.info(f"Final FPS: {final_fps:.2f}")

        except Exception as e:
            logger.error(f"Error during frame acquisition: {e}")
        finally:
            logger.info("Frame acquisition process is terminating.")
            self.shared_buffer.cleanup()
