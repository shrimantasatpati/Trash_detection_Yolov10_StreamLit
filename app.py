# import streamlit as st
# import torch
# from ultralytics import YOLO
# from PIL import Image
# import supervision as sv
# bounding_box_annotator = sv.BoundingBoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # Setting page layout
# st.set_page_config(
#     page_title="Trash detection - YOLOv10",
#     page_icon="https://csc.edu.vn/favicon.ico",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Main page heading
# st.title("Trash detection - YOLOv10")

# # Sidebar
# st.sidebar.header("ML Model Config")

# # confidence = float(st.sidebar.slider(
# #     "Select Model Confidence", 5, 100, 20)) / 100
# confidence = st.sidebar.slider(
#     "Select Model Confidence", 
#     min_value=0.0,
#     max_value=1.0,
#     value=0.2,  # Default value
#     step=0.05
# )
# model_choice = st.sidebar.radio("Select Model", ["Model 1", "Model 2"])
# # max_det = st.sidebar.slider(
# #     "Select maximum number of detected objects", 5, 1000, 20)
# # show_labels = st.sidebar.radio("Show Labels", [True, False], index=0)
# # show_boxes = st.sidebar.radio("Show Boxes", [True, False], index=0)

# # Load YOLO model
# @st.cache_resource()
# # def main_model():
# #     model = YOLO('best.pt')
# #     return model
# def main_model(model_choice):
#     if model_choice == "Model 1":
#         model = YOLO('best.pt')
#     else:
#         model = YOLO('best_yolov10_garbage_classification.pt')
#     return model

# # File uploader widget
# uploaded_file = st.file_uploader("Choose an image...",
#                             type=['jpg', 'jpeg', 'png', "bmp", "webp"]
#                             )

# col1, col2 = st.columns(2)

# with col1:
#     if uploaded_file is not None:
#         uploaded_image = Image.open(uploaded_file)
#         st.image(uploaded_image, caption="Uploaded Image",
#                 use_column_width=True)
#     else:
#         st.info("Please upload an image.")

# with col2:
#     if st.sidebar.button('Detect') and uploaded_file is not None:
#         try:
#             # Make predictions on the uploaded image
#             with torch.no_grad():
#                 model = main_model(model_choice)
#                 results = model(
#                     task="detect",
#                     source=uploaded_image,
#                     # max_det=max_det,
#                     conf=confidence,
#                     # show_labels=show_labels,
#                     # show_boxes=show_boxes,
#                     save=False,
#                     device="cpu"
#                 )
#                 boxes = results[0].boxes
#                 res_plotted = results[0].plot()[:, :, ::-1]
#                 st.image(res_plotted, caption='Detect Image',
#                          use_column_width=True)

                                
#                 # Count the number of objects for each detected class
#                 class_counts = {}
#                 for cls in boxes.cls:
#                     class_name = model.names[int(cls)]
#                     if class_name in class_counts:
#                         class_counts[class_name] += 1
#                     else:
#                         class_counts[class_name] = 1
                
                                
#                 # Prepare data for the table
#                 table_data = [{"Class": class_name, "Count": count} for class_name, count in class_counts.items()]
                
#                 # Display the table
#                 st.write("Number of objects for each detected class:")
#                 st.table(table_data)

#                 # results = model(source=uploaded_image, conf=0.25)[0]
#                 # results[0].show()
#                 # detections = sv.Detections.from_ultralytics(results[0])
                
#                 # annotated_image = bounding_box_annotator.annotate(
#                 #     scene=uploaded_image, detections=detections)
#                 # annotated_image = label_annotator.annotate(
#                 #     scene=annotated_image, detections=detections)
                
#                 # sv.plot_image(annotated_image)
                

#             with st.expander("Detection Results"):
#                 for box in boxes:
#                     st.write(box.data)

#         except Exception as ex:
#             st.exception(ex)
#     else:
#         st.error("Please select a valid source type!")

# import streamlit as st
# import torch
# from ultralytics import YOLO
# from PIL import Image
# import supervision as sv
# bounding_box_annotator = sv.BoundingBoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # Setting page layout
# st.set_page_config(
#     page_title="Trash detection - YOLOv10",
#     page_icon="https://csc.edu.vn/favicon.ico",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Main page heading
# st.title("Trash detection - YOLOv10")

# # Sidebar
# st.sidebar.header("ML Model Config")

# # confidence = float(st.sidebar.slider(
# #     "Select Model Confidence", 5, 100, 20)) / 100
# confidence = st.sidebar.slider(
#     "Select Model Confidence", 
#     min_value=0.0,
#     max_value=1.0,
#     value=0.2,  # Default value
#     step=0.05
# )
# model_choice = st.sidebar.radio("Select Model", ["Model 1", "Model 2"])
# # max_det = st.sidebar.slider(
# #     "Select maximum number of detected objects", 5, 1000, 20)
# # show_labels = st.sidebar.radio("Show Labels", [True, False], index=0)
# # show_boxes = st.sidebar.radio("Show Boxes", [True, False], index=0)

# # Load YOLO model
# @st.cache_resource()
# def main_model():
#     model = YOLO('best.pt')
#     return model
# # File uploader widget
# uploaded_file = st.file_uploader("Choose an image...",
#                             type=['jpg', 'jpeg', 'png', "bmp", "webp"]
#                             )

# col1, col2 = st.columns(2)

# with col1:
#     if uploaded_file is not None:
#         uploaded_image = Image.open(uploaded_file)
#         st.image(uploaded_image, caption="Uploaded Image",
#                 use_column_width=True)
#     else:
#         st.info("Please upload an image.")

# with col2:
#     if st.sidebar.button('Detect') and uploaded_file is not None:
#         try:
#             # Make predictions on the uploaded image
#             with torch.no_grad():
#                 model = main_model(model_choice)
#                 results = model(
#                     task="detect",
#                     source=uploaded_image,
#                     # max_det=max_det,
#                     conf=confidence,
#                     # show_labels=show_labels,
#                     # show_boxes=show_boxes,
#                     save=False,
#                     device="cpu"
#                 )
#                 boxes = results[0].boxes
#                 res_plotted = results[0].plot()[:, :, ::-1]
#                 st.image(res_plotted, caption='Detect Image',
#                          use_column_width=True)

                                
#                 # Count the number of objects for each detected class
#                 class_counts = {}
#                 for cls in boxes.cls:
#                     class_name = model.names[int(cls)]
#                     if class_name in class_counts:
#                         class_counts[class_name] += 1
#                     else:
#                         class_counts[class_name] = 1
                
                                
#                 # Prepare data for the table
#                 table_data = [{"Class": class_name, "Count": count} for class_name, count in class_counts.items()]
                
#                 # Display the table
#                 st.write("Number of objects for each detected class:")
#                 st.table(table_data)
#             with st.expander("Detection Results"):
#                 for box in boxes:
#                     st.write(box.data)

#         except Exception as ex:
#             st.exception(ex)
#     else:
#         st.error("Please select a valid source type!")


import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import supervision as sv
import cv2
import tempfile
import os
from pathlib import Path

# Initialize annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Setting page layout
st.set_page_config(
    page_title="Trash detection - YOLOv10",
    page_icon="https://csc.edu.vn/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Trash detection - YOLOv10")

# Sidebar
st.sidebar.header("ML Model Config")

# Model configuration
confidence = st.sidebar.slider(
    "Select Model Confidence", 
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

# Source type selection
source_type = st.sidebar.radio("Select Source Type", ["Image", "Video"])

model_choice = st.sidebar.radio("Select Model", ["Model"])

# Load YOLO model
@st.cache_resource
def main_model():
    model = YOLO('best.pt')
    return model

# Process single image
def process_image(model, image, confidence):
    with torch.no_grad():
        results = model(
            source=image,
            conf=confidence,
            device="cpu"
        )
        return results[0]

# Process video frame
def process_frame(model, frame, confidence):
    with torch.no_grad():
        results = model(
            source=frame,
            conf=confidence,
            device="cpu"
        )
        return results[0]

# Display detection results
def display_detection_results(results):
    boxes = results.boxes
    res_plotted = results.plot()[:, :, ::-1]
    st.image(res_plotted, caption='Detected Objects', use_column_width=True)
    
    # Count objects
    class_counts = {}
    for cls in boxes.cls:
        class_name = model.names[int(cls)]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    # Display counts
    table_data = [{"Class": class_name, "Count": count} 
                 for class_name, count in class_counts.items()]
    st.write("Number of objects for each detected class:")
    st.table(table_data)
    
    # Show detection details
    with st.expander("Detection Results"):
        for box in boxes:
            st.write(box.data)

# Main application logic
if source_type == "Image":
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', "bmp", "webp"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image",
                    use_column_width=True)
        else:
            st.info("Please upload an image.")
            
    with col2:
        if st.sidebar.button('Detect') and uploaded_file is not None:
            try:
                model = main_model()
                results = process_image(model, uploaded_image, confidence)
                display_detection_results(results)
            except Exception as ex:
                st.exception(ex)
        else:
            st.error("Please upload an image first!")

else:  # Video processing
    uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        # Display uploaded video
        st.video(uploaded_file)
        
        if st.sidebar.button('Detect'):
            try:
                model = main_model()
                
                # Read video
                cap = cv2.VideoCapture(tfile.name)
                
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Create temporary output file
                output_path = str(Path(tempfile.mkdtemp()) / "output.mp4")
                
                # Initialize video writer
                out = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width, height)
                )
                
                # Process video frames
                progress_text = "Processing video..."
                progress_bar = st.progress(0)
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    results = process_frame(model, frame, confidence)
                    processed_frame = results.plot()
                    
                    # Write processed frame
                    out.write(processed_frame)
                    
                    # Update progress
                    frame_count += 1
                    progress_bar.progress(frame_count / total_frames)
                
                # Release resources
                cap.release()
                out.release()
                
                # Display processed video
                st.success("Video processing complete!")
                st.video(output_path)
                
                # Clean up temporary files
                os.unlink(tfile.name)
                os.unlink(output_path)
                
            except Exception as ex:
                st.exception(ex)
    else:
        st.info("Please upload a video.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Shrimanta Satpati")
