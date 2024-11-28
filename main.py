import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from config import config
from detection_engine import DetectionEngine
import logging
from gtts import gTTS
from playsound import playsound
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class PerformanceMetrics:
    fps: float = 0.0
    processing_time: float = 0.0
    frame_count: int = 0
    detection_count: int = 0
    
    def update(self, frame_time: float, num_detections: int):
        self.frame_count += 1
        self.processing_time += frame_time
        self.detection_count += num_detections
        self.fps = self.frame_count / max(self.processing_time, 0.001)

class ApplicationState:
    def __init__(self):
        self.running: bool = True
        self.paused: bool = False
        self.show_debug: bool = False
        self.audio_enabled: bool = True
        self.metrics = PerformanceMetrics()

class Application:
    def __init__(self):
        """Initialize the application."""
        self.state = ApplicationState()
        self.detection_engine = DetectionEngine()
        self.setup_queues()
        self.setup_windows()
        self.frame_id = 0
        self.prev_frame = None
        self.prev_detections = None
        self.last_audio_time = 0
        logging.info("Application initialized")

    def setup_queues(self):
        """Initialize processing queues."""
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.audio_queue = queue.Queue()

    def setup_windows(self):
        """Setup display windows."""
        cv2.namedWindow('Main Feed', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Main Feed', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.show_debug = not self.state.show_debug
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.state.paused = not self.state.paused

    def speak(self, text: str):
        """Text-to-speech conversion."""
        if not self.state.audio_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_audio_time < 5:  # Minimum 5 seconds between audio
            return
            
        try:
            filename = "temp_speech.mp3"
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
            self.last_audio_time = current_time
        except Exception as e:
            logging.error(f"Error in speech synthesis: {e}")

    @contextmanager
    def video_capture(self):
        """Context manager for video capture."""
        cap = cv2.VideoCapture(0)
        try:
            if not cap.isOpened():
                raise RuntimeError("Could not open video capture")
            yield cap
        finally:
            cap.release()

    @contextmanager
    def video_writer(self, frame_size):
        """Context manager for video writer."""
        writer = cv2.VideoWriter(
            str(config.output_video_path),
            config.VIDEO_CODEC,
            config.FPS,
            frame_size
        )
        try:
            yield writer
        finally:
            writer.release()

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth using motion detection."""
        if self.prev_frame is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
            
        try:
            # Convert frames to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            
            # Convert flow to magnitude
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            depth_map = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
            return depth_map.astype(np.uint8)
        except Exception as e:
            logging.error(f"Error estimating depth: {e}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    def add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add information overlay to frame."""
        try:
            metrics = self.state.metrics
            info = [
                f"Frame: {self.frame_id}",
                f"FPS: {metrics.fps:.1f}",
                f"Detections: {metrics.detection_count}",
                f"Processing Time: {metrics.processing_time*1000:.1f}ms",
                f"Status: {'Paused' if self.state.paused else 'Running'}",
                f"Audio: {'Enabled' if self.state.audio_enabled else 'Disabled'}"
            ]
            
            # Add dark background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (10, 10),
                (300, 40 + len(info) * 30),
                config.COLORS['background'],
                -1
            )
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            # Add text
            for i, text in enumerate(info):
                cv2.putText(
                    frame,
                    text,
                    (20, 40 + i * 30),
                    config.FONT,
                    config.FONT_SCALE,
                    config.COLORS['text'],
                    config.FONT_THICKNESS
                )
            
            return frame
        except Exception as e:
            logging.error(f"Error adding overlay: {e}")
            return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Process a single frame."""
        try:
            start_time = time.time()
            
            # Get detections
            detections, annotated_frame = self.detection_engine.detect_and_track(frame)
            
            # Estimate depth if needed
            if self.state.show_debug:
                depth_map = self.estimate_depth(frame)
                cv2.imshow('Depth Map', depth_map)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.state.metrics.update(processing_time, len(detections))
            
            # Add information overlay
            if len(detections) > 0:
                # Audio alert for significant detections
                self.audio_queue.put(f"Detected {len(detections)} objects")
                
            annotated_frame = self.add_overlay(annotated_frame)
            
            # Update previous frame
            self.prev_frame = frame.copy()
            self.prev_detections = detections
            
            return annotated_frame, len(detections)
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame, 0

    def audio_worker(self):
        """Worker thread for audio processing."""
        while self.state.running:
            try:
                text = self.audio_queue.get(timeout=1.0)
                if text:
                    self.speak(text)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio worker: {e}")

    def handle_keyboard(self) -> bool:
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.state.running = False
            return False
        elif key == ord('p'):
            self.state.paused = not self.state.paused
        elif key == ord('d'):
            self.state.show_debug = not self.state.show_debug
        elif key == ord('a'):
            self.state.audio_enabled = not self.state.audio_enabled
        return True

    def run(self):
        """Main application loop."""
        logging.info("Starting application...")
        
        # Start audio worker thread
        audio_thread = threading.Thread(target=self.audio_worker)
        audio_thread.daemon = True
        audio_thread.start()
        
        try:
            with self.video_capture() as cap:
                frame_size = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                
                with self.video_writer(frame_size) as writer:
                    while self.state.running:
                        if not self.state.paused:
                            ret, frame = cap.read()
                            if not ret:
                                logging.warning("Failed to read frame")
                                continue
                            
                            self.frame_id += 1
                            
                            # Process frame
                            annotated_frame, num_detections = self.process_frame(frame)
                            
                            # Display and record
                            cv2.imshow('Main Feed', annotated_frame)
                            writer.write(annotated_frame)
                        
                        # Handle keyboard input
                        if not self.handle_keyboard():
                            break
                            
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up...")
        self.state.running = False
        cv2.destroyAllWindows()
        
        # Print final statistics
        metrics = self.state.metrics
        logging.info(f"""
        Final Statistics:
        - Total Frames: {metrics.frame_count}
        - Average FPS: {metrics.fps:.2f}
        - Total Detections: {metrics.detection_count}
        - Total Processing Time: {metrics.processing_time:.2f}s
        - Average Detections per Frame: {metrics.detection_count/max(metrics.frame_count, 1):.2f}
        """)

        

if __name__ == "__main__":
    app = Application()
    try:
        app.run()
    except KeyboardInterrupt:
        logging.info("Application terminated by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        app.cleanup()
