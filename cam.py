import cv2
import numpy as np
import time
import threading
import queue
from gtts import gTTS
from playsound import playsound
import os
import sys

from config import (
    weight_file, ALL_OBJECTS, fps, 
    output_video_filename, fourcc, YOLO_INTERVAL, GPT_INTERVAL
)
from yolo_model import initialize_model, predict
from utils import calculate_movements
from gpt_interface import GPT_annotation
from video_processing import process_frame  # Removed annotate_frame from import

def speak(text, filename="speech.mp3"):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        playsound(filename)
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"Error in speech synthesis: {e}")

def add_danger_assessment(frame, danger_assessment, last_navigation):
    try:
        if danger_assessment is None:
            danger_score = 0
            reason = 'No danger detected'
            navigation = 'Path clear. Move forward.'
        else:
            danger_score = danger_assessment.get('danger_score', 0)
            reason = danger_assessment.get('reason', 'No danger detected')
            navigation = danger_assessment.get('navigation', 'Path clear. Move forward.')
        
        color = (0, int(255*(1-danger_score)), int(255*danger_score))
        
        cv2.rectangle(frame, (0, frame.shape[0] - 90), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Danger: {danger_score:.1f} | {reason}", (10, frame.shape[0] - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Navigation: {navigation}", (10, frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if navigation != last_navigation:
            threading.Thread(target=speak, args=(navigation,)).start()
        
        return frame, navigation
    except Exception as e:
        print(f"Error in danger assessment: {e}")
        return frame, last_navigation

def gpt_worker(input_queue, output_queue):
    while True:
        try:
            categorized_detections = input_queue.get()
            if categorized_detections is None:
                break
            danger_assessment, _, _ = GPT_annotation(categorized_detections)
            output_queue.put(danger_assessment)
        except Exception as e:
            print(f"Error in GPT worker: {e}")
            output_queue.put(None)

def simulate_depth(frame, prev_frame):
    try:
        if prev_frame is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Convert frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        frame_diff = cv2.absdiff(gray, prev_gray)
        
        # Threshold the difference
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply some morphological operations to remove noise
        kernel = np.ones((5,5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        return motion_mask
    except Exception as e:
        print(f"Error in depth simulation: {e}")
        return np.zeros(frame.shape[:2], dtype=np.uint8)

def mouse_event_handler(event, x, y, flags, param):
    try:
        if event == cv2.EVENT_MOUSEMOVE:
            depth_value = param['depth_frame'][y, x]
            param['text'] = f"Position: ({x}, {y}), Motion: {depth_value}"
    except Exception as e:
        print(f"Error in mouse handler: {e}")
        param['text'] = ""

def main():
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("Initializing YOLO model...")
        model = initialize_model(weight_file)
        
        print("Setting up video writer...")
        output_video = cv2.VideoWriter(
            output_video_filename, 
            fourcc, 
            fps, 
            (frame_width, frame_height)
        )

        # Initialize variables
        frame_id = 0
        prev_frame = None
        start_time = time.time()
        danger_assessment = None
        last_yolo_results = None
        last_navigation = ""
        last_fps_update = time.time()
        fps_display = 0

        # Initialize queues and thread for GPT processing
        print("Setting up GPT processing...")
        gpt_input_queue = queue.Queue()
        gpt_output_queue = queue.Queue()
        gpt_thread = threading.Thread(target=gpt_worker, args=(gpt_input_queue, gpt_output_queue))
        gpt_thread.daemon = True  # Make thread daemon so it exits when main program exits
        gpt_thread.start()

        # Set up display window
        cv2.namedWindow('Camera Feed')
        text_info = {'text': '', 'depth_frame': None}
        cv2.setMouseCallback('Camera Feed', mouse_event_handler, text_info)

        print("Starting main loop...")
        while True:
            loop_start = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Retrying...")
                time.sleep(0.1)  # Add small delay before retry
                continue  # Try to get the next frame instead of breaking

            frame_id += 1

            # Update FPS display every second
            if time.time() - last_fps_update >= 1.0:
                fps_display = frame_id / (time.time() - start_time)
                last_fps_update = time.time()

            try:
                # Generate depth information
                depth_frame = simulate_depth(frame, prev_frame)
                text_info['depth_frame'] = depth_frame

                # YOLO detection
                if frame_id % YOLO_INTERVAL == 0:
                    try:
                        results = predict(model, frame)
                        if results is not None:
                            last_yolo_results = results
                    except Exception as e:
                        print(f"Error in YOLO prediction: {e}")
                        results = last_yolo_results
                else:
                    results = last_yolo_results

                # Calculate movements
                movements = {}
                if results is not None and last_yolo_results is not None and prev_frame is not None:
                    try:
                        movements = calculate_movements(last_yolo_results, results)
                    except Exception as e:
                        print(f"Error calculating movements: {e}")

                # Process frame
                try:
                    annotated_frame, categorized_detections = process_frame(frame, results, movements, frame_id)
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    annotated_frame = frame.copy()
                    categorized_detections = {}

                # GPT processing
                if frame_id % GPT_INTERVAL == 0 and categorized_detections:
                    gpt_input_queue.put(categorized_detections)

                while not gpt_output_queue.empty():
                    danger_assessment = gpt_output_queue.get()

                # Add danger assessment
                annotated_frame, last_navigation = add_danger_assessment(
                    annotated_frame, danger_assessment, last_navigation
                )

                # Add FPS and other information
                cv2.putText(annotated_frame, f'Frame: {frame_id} FPS: {fps_display:.1f}', 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, text_info['text'], 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display and record
                cv2.imshow('Camera Feed', annotated_frame)
                cv2.imshow('Simulated Depth', depth_frame)
                output_video.write(annotated_frame)

                # Store current frame for next iteration
                prev_frame = frame.copy()

                # Handle quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit command received.")
                    break

                # Add delay if processing is too fast
                elapsed = time.time() - loop_start
                if elapsed < 1.0/fps:
                    time.sleep(1.0/fps - elapsed)

            except Exception as e:
                print(f"Error in main loop: {e}")
                continue  # Continue to next frame instead of breaking

    except Exception as e:
        print(f"Critical error in main function: {e}")
    
    finally:
        print("Cleaning up...")
        # Cleanup
        try:
            gpt_input_queue.put(None)  # Signal GPT thread to stop
            gpt_thread.join(timeout=1.0)  # Wait for GPT thread with timeout
            cap.release()
            output_video.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {e}")

        # Print statistics
        end_time = time.time()
        total_time = end_time - start_time
        print('Total time:', total_time)
        print('Average FPS:', frame_id / total_time)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
