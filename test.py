import cv2
import dv
import dv.io
import dv.processing
import signal
import time

aedat4_path = r'C:\Users\damig\MN\Projekt_alternatywny\ujecie1_kz\atak z użyciem przedmiotu\atak-2022_11_16_11_55_49.aedat4'

def handle_shutdown(signum, frame):
    global global_shutdown
    global_shutdown = True

def handle_interval(events):
    global frame_count
    # Pass event and generate a frame
    accumulator.accept(events)
    frame = accumulator.generate_frame()

    # Save the accumulated frame to a file with an incrementing name
    file_name = f"accumulated_frame_{frame_count}.jpg"
    cv2.imwrite(file_name, frame.image)
    frame_count += 1

# Install signal handlers for a clean shutdown
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Construct the reader
reader = dv.io.MonoCameraRecording(aedat4_path)

# Placeholders for stream names
frame_stream = ""
event_stream = ""

# Find streams with compatible types from the list of all available streams
for name in reader.get_stream_names():
    if reader.is_stream_of_data_type(name, dv.Frame) and not frame_stream:
        frame_stream = name
    elif reader.is_stream_of_data_type(name, dv.EventPacket) and not event_stream:
        event_stream = name

# Named variables to hold the availability of streams
frames_available = bool(frame_stream)
events_available = bool(event_stream)

# Check whether at least one of the streams is available
if not frames_available and not events_available:
    raise ValueError(f"Aedat4 player requires a file with at least event or frame stream available: {aedat4_path}")

 # Create an accumulator for the reader
accumulator = dv.Accumulator(reader.get_event_resolution())

# Initialize event stream slicer
slicer = dv.EventStreamSlicer()
slicer.do_every_time_interval(1.0, handle_interval)

# Buffer to store last frame
last_frame = dv.Frame()
last_frame.timestamp = -1

# Initialize frame count
frame_count = 0

# Main reading loop
while not global_shutdown and reader.is_running():
    # Read frames if available
    if frames_available:
        frame = reader.read_frame(frame_stream)
        if frame.timestamp != last_frame.timestamp:
            # Show the frame

            cv2.imshow("AEDAT4 Player - Frames", frame.image)
            cv2.waitKey(2)

            # Update the last frame
            last_frame = frame
    elif events_available:
        # Sleep for a short period to match the slicer interval
        time.sleep(0.050)