import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import supervision as sv
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

# confidence = float(st.sidebar.slider(
#     "Select Model Confidence", 5, 100, 20)) / 100
confidence = st.sidebar.slider(
    "Select Model Confidence", 
    min_value=0.0,
    max_value=1.0,
    value=0.2,  # Default value
    step=0.05
)
model_choice = st.sidebar.radio("Select Model", ["Model 1", "Model 2"])
# max_det = st.sidebar.slider(
#     "Select maximum number of detected objects", 5, 1000, 20)
# show_labels = st.sidebar.radio("Show Labels", [True, False], index=0)
# show_boxes = st.sidebar.radio("Show Boxes", [True, False], index=0)

# Load YOLO model
@st.cache_resource()
# def main_model():
#     model = YOLO('best.pt')
#     return model
def main_model(model_choice):
    if model_choice == "Model 1":
        model = YOLO('best.pt')
    else:
        model = YOLO('best_yolov10_garbage_classification.pt')
    return model

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...",
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
            # Make predictions on the uploaded image
            with torch.no_grad():
                model = main_model(model_choice)
                results = model(
                    task="detect",
                    source=uploaded_image,
                    # max_det=max_det,
                    conf=confidence,
                    # show_labels=show_labels,
                    # show_boxes=show_boxes,
                    save=False,
                    device="cpu"
                )
                boxes = results[0].boxes
                res_plotted = results[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detect Image',
                         use_column_width=True)

                                
                # Count the number of objects for each detected class
                class_counts = {}
                for cls in boxes.cls:
                    class_name = model.names[int(cls)]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                
                                
                # Prepare data for the table
                table_data = [{"Class": class_name, "Count": count} for class_name, count in class_counts.items()]
                
                # Display the table
                st.write("Number of objects for each detected class:")
                st.table(table_data)

                # results = model(source=uploaded_image, conf=0.25)[0]
                # results[0].show()
                # detections = sv.Detections.from_ultralytics(results[0])
                
                # annotated_image = bounding_box_annotator.annotate(
                #     scene=uploaded_image, detections=detections)
                # annotated_image = label_annotator.annotate(
                #     scene=annotated_image, detections=detections)
                
                # sv.plot_image(annotated_image)
                

            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.data)

        except Exception as ex:
            st.exception(ex)
    else:
        st.error("Please select a valid source type!")
