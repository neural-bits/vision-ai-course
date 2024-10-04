import multiprocessing as mp
from multiprocessing import Process

import cv2
from core.buffer import SharedBuffer
from inference import InferenceHandler
from video import FrameAcquisition

# Global variables for processes and buffer
shared_buffer = None
acquisition_process = None

# cv2.setNumThreads(1)


def main():
    global shared_buffer, acquisition_process, inference_process
    mp.set_start_method("fork")

    # Define shared buffer
    frame_shape = (720, 1280, 3)
    buffer_size = 5
    shared_buffer = SharedBuffer(frame_shape, buffer_size)

    # Start acquisition and inference processes
    acquisition = FrameAcquisition(
        shared_buffer,
        "output_video_hd.mp4",
    )
    infer = InferenceHandler(shared_buffer)

    acquisition_process = Process(target=acquisition.acquire_frames)
    inference_process1 = Process(target=infer.execute)

    acquisition_process.start()
    inference_process1.start()

    acquisition_process.join()
    inference_process1.join()

    # When the application shuts down, clean up resources
    if acquisition_process and acquisition_process.is_alive():
        acquisition_process.terminate()
        acquisition_process.join()

    # if inference_process and inference_process.is_alive():
    # inference_process.terminate()
    # inference_process.join()

    shared_buffer.cleanup()


if __name__ == "__main__":
    main()
