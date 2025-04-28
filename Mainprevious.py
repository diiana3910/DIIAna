import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import defaultdict
import pandas as pd
import tempfile
import xml.etree.ElementTree as ET
from skimage.morphology import skeletonize

# Function to calculate crack properties using Approach 2 (Skeletonization)
def calculate_crack_properties_approach2(mask, conversion_factor=3.5):
    # Skeletonize the mask
    skeleton = skeletonize(mask // 255)
    
    # Calculate length as the sum of the skeleton
    length_px = np.sum(skeleton)
    length_mm = length_px * conversion_factor
    
    # Calculate area using OpenCV
    area_px = np.sum(mask > 0)
    area_mm = area_px * (conversion_factor ** 2)
    
    # Approximate width as area / length
    width_px = area_px / length_px if length_px > 0 else 0
    width_mm = width_px * conversion_factor
    
    return length_mm, width_mm, area_mm

# Function to calculate severity based on the width
def calculate_severity(width):
    if 0 < width <= 3:
        return "Low"
    elif 3 < width <= 6:
        return "Medium"
    else:
        return "High"

# Function to draw annotations on the image
def draw_annotations(image, cracks):
    annotated_image = image.copy()

    class_colors = {
        'Longitudinal crack': (0, 0, 255),  # Red
        'Oblique crack': (0, 0, 0),  # Black
        'Alligator crack': (255, 0, 0),  # Blue
        'Pothole': (0, 255, 0),  # Green
        'Repair': (255, 255, 255),  # White
    }

    for idx, crack in enumerate(cracks):
        crack_type = crack['name']
        color_for_crack = class_colors.get(crack_type)
        points = np.array(crack['points'], dtype=np.int32)
        cv2.polylines(annotated_image, [points], isClosed=True, color=color_for_crack, thickness=3)
        # Label the crack with its name
        centroid = np.mean(points, axis=0).astype(int)
        cv2.putText(
            annotated_image,
            f"{chr(65 + idx)}",  
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,  # Increased font size
            (0, 0, 0),  # Black color
            3,  # Thickness
            lineType=cv2.LINE_AA,
        )
    return annotated_image

# Function to extract frames from a video
def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    success, image = cap.read()
    while success:
        if frame_count % frame_rate == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(image))
        success, image = cap.read()
        frame_count += 1
    cap.release()
    return frames

# Function to process a frame with the YOLO model
def process_frame_with_model(frame, model, selected_objects, min_confidence, class_colors):
    img = np.array(frame)
    result = model(img)
    distress_counts, distress_areas = process_results(result, img, model, selected_objects, min_confidence, class_colors)
    return img, distress_counts, distress_areas

# Function to process YOLO results
def process_results(result, img, model, selected_objects, min_confidence, class_colors):
    distress_counts = defaultdict(int)
    distress_areas = defaultdict(float)

    for detection in result[0].boxes.data:
        x0, y0 = (int(detection[0]), int(detection[1]))
        x1, y1 = (int(detection[2]), int(detection[3]))
        score = round(float(detection[4]), 2)
        cls = int(detection[5])
        object_name = model.names[cls]
        label = f'{object_name} {score}'

        if object_name in selected_objects and score > min_confidence:
            color = class_colors.get(object_name, (255, 255, 255))  
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 4)
            cv2.putText(img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

            distress_counts[object_name] += 1
            distress_area = (x1 - x0) * (y1 - y0)
            distress_areas[object_name] += distress_area

    return distress_counts, distress_areas

# Function to display quantification results
def display_quantification_results(distress_counts, distress_areas, img_height, img_width):
    total_image_area = img_height * img_width
    total_distress_area = sum(distress_areas.values())
    total_distress_area_percentage = (total_distress_area / total_image_area) * 100

    data = []
    for distress_type, count in distress_counts.items():
        area_percentage = (distress_areas[distress_type] / total_image_area) * 100
        data.append({
            "Distress Type": distress_type,
            "Count": count,
            "% of Distressed Area": f"{area_percentage:.2f}%"
        })

    df = pd.DataFrame(data)
    total_row = pd.DataFrame({
        "Distress Type": ["Total"],
        "Count": [sum(distress_counts.values())],
        "% of Distressed Area": [f"{total_distress_area_percentage:.2f}%"]
    })
    df = pd.concat([df, total_row], ignore_index=True)

    st.write("Quantification Results:")
    st.dataframe(data=df, use_container_width=True, hide_index=True)

# Function to display identification results
def display_results(img, distress_counts, distress_areas, img_height, img_width):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    st.image(img_pil, caption="Detected Distresses!", use_container_width=True)

    total_image_area = img_height * img_width
    total_distress_area = sum(distress_areas.values())
    total_distress_area_percentage = (total_distress_area / total_image_area) * 100

    data = []
    for distress_type, count in distress_counts.items():
        area_percentage = (distress_areas[distress_type] / total_image_area) * 100
        data.append({
            "Distress Type": distress_type,
            "Count": count,
            "% of Distressed Area": f"{area_percentage:.2f}%"
        })

    df = pd.DataFrame(data)
    total_row = pd.DataFrame({
        "Distress Type": ["Total"],
        "Count": [sum(distress_counts.values())],
        "% of Distressed Area": [f"{total_distress_area_percentage:.2f}%"]
    })
    df = pd.concat([df, total_row], ignore_index=True)

    st.write("Identification Results:")
    st.dataframe(data=df, use_container_width=True, hide_index=True)

# Main function for pavement analysis
def pavement_analysis(title):
    st.title(title)
    detection_model = YOLO('300_Epochs_V8_Extra_Large_Detect.pt')
    segmentation_model = YOLO('445_Epochs_V8_Medium_Segment.pt')
    object_names = list(detection_model.names.values())

    class_colors = {
        'Longitudinal crack': (0, 0, 255),  # Red
        'Oblique crack': (0, 0, 0),  # Black
        'Alligator crack': (255, 0, 0),  # Blue
        'Pothole': (0, 255, 0),  # Green
        'Repair': (255, 255, 255),  # White
    }

    if "selected_objects" not in st.session_state:
        st.session_state.selected_objects = []

    if "active_module" not in st.session_state:
        st.session_state.active_module = None  

    if "identification_results" not in st.session_state:
        st.session_state.identification_results = None  

    if "quantification_results" not in st.session_state:
        st.session_state.quantification_results = None  

    with st.form("my_form"):
        col_img, col_vid = st.columns(2)
        
        with col_img:
            uploaded_image = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
        
        with col_vid:
            uploaded_video = st.file_uploader('Upload Video', type=['mp4', 'avi', 'mov'])
        
        col1, col2 = st.columns([0.7, 0.3])
        with col2:
            if st.form_submit_button("Select All Distresses"):
                st.session_state.selected_objects = object_names  
        
        with col1:
            selected_objects = st.multiselect(
                'Choose the type of distresses to detect',
                object_names,
                default=st.session_state.selected_objects
            )
        
        min_confidence = st.slider('Confidence score', 0.0, 1.0)
        submit_button = st.form_submit_button(label='Submit')

    st.session_state.selected_objects = selected_objects

    # Add buttons for Identification and Quantification modules
    col1, col2 = st.columns([0.74, 0.26])
    with col1:
        if st.button("Identification Module"):
            st.session_state.active_module = "identification"
    with col2:
        if st.button("Quantification Module"):
            st.session_state.active_module = "quantification"

    # Process the form submission
    if submit_button:
        if uploaded_image is not None and uploaded_video is None:
            file_bytes = uploaded_image.read()
            n_parr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(n_parr, cv2.IMREAD_COLOR)
            img_height, img_width, _ = img.shape

            with st.spinner('Analyzing Image...'):
                result = detection_model(img)

            distress_counts, distress_areas = process_results(result, img, detection_model, selected_objects, min_confidence, class_colors)
            st.session_state.identification_results = (img, distress_counts, distress_areas, img_height, img_width)

        elif uploaded_video is not None and uploaded_image is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_video.read())
                temp_video_path = temp_video.name
            
            frames = extract_frames(temp_video_path, frame_rate=30)  
            st.session_state.identification_results = frames  # Store frames for video processing

    # Display results based on the active module
    if st.session_state.active_module == "identification":
        if st.session_state.identification_results is not None:
            if isinstance(st.session_state.identification_results, tuple):  # Image results
                img, distress_counts, distress_areas, img_height, img_width = st.session_state.identification_results
                display_results(img, distress_counts, distress_areas, img_height, img_width)
            else:  # Video results
                for frame in st.session_state.identification_results:
                    processed_frame, distress_counts, distress_areas = process_frame_with_model(frame, detection_model, selected_objects, min_confidence, class_colors)
                    display_results(processed_frame, distress_counts, distress_areas, processed_frame.shape[0], processed_frame.shape[1])

    elif st.session_state.active_module == "quantification":
        if st.session_state.identification_results is not None:
            if isinstance(st.session_state.identification_results, tuple):  # Image results
                img, distress_counts, distress_areas, img_height, img_width = st.session_state.identification_results

                # Process the image with the segmentation model
                with st.spinner('Segmenting Image...'):
                    segmentation_result = segmentation_model(img)

                # Extract masks and bounding boxes from the segmentation result
                masks = segmentation_result[0].masks.data.cpu().numpy() if segmentation_result[0].masks else None
                detection_boxes = segmentation_result[0].boxes.xyxy.cpu().numpy() if segmentation_result[0].boxes else None

                if masks is not None and detection_boxes is not None:
                    combined_results = []
                    conversion_factor = 3.5  # 0.35 cm/px * 10 mm/cm = 3.5 mm/px

                    # Draw annotations on the image
                    annotated_image = img.copy()

                    for idx, (mask, box) in enumerate(zip(masks, detection_boxes)):
                        mask = (mask * 255).astype(np.uint8)  # Convert mask to binary image
                        length_mm, width_mm, area_mm = calculate_crack_properties_approach2(mask, conversion_factor)
                        severity = calculate_severity(width_mm)

                        # Get the class name from the segmentation model's names dictionary
                        class_id = segmentation_result[0].boxes.cls[idx].item()  
                        if class_id in segmentation_result[0].names:  
                            class_name = segmentation_result[0].names[class_id]
                        else:
                            class_name = f"Unknown Class {class_id}"  

                        # Calculate the centroid of the bounding box for placing the identification code
                        x0, y0, x1, y1 = box
                        centroid = (int((x0 + x1) / 2), int((y0 + y1) / 2))

                        # Draw the identification code (A, B, C, etc.) on the image
                        cv2.putText(
                            annotated_image,
                            f"{chr(65 + idx)}",  
                            centroid,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,  
                            (0, 0, 0),  
                            3,  
                            lineType=cv2.LINE_AA,
                        )

                        combined_results.append({
                            'S.No.': idx + 1,  # Serial number starting from 1
                            'Type of Crack': class_name,
                            'Identification Code': chr(65 + idx),
                            'Length (mm)': length_mm,
                            'Width (mm)': width_mm,
                            'Area (mm²)': area_mm,
                            'Severity': severity
                        })

                    # Display the annotated image
                    st.image(annotated_image, caption="Annotated Image with Identification Codes", use_container_width=True)

                    # Display results in a table
                    st.write("### Crack Properties")
                    df = pd.DataFrame(combined_results)
                    st.table(df.set_index('S.No.')) 
                else:
                    st.error("No cracks detected in the image.")
            else:  # Video results
                for frame in st.session_state.identification_results:
                    # Process the frame with the segmentation model
                    with st.spinner('Segmenting Frame...'):
                        segmentation_result = segmentation_model(np.array(frame))

                    # Extract masks and bounding boxes from the segmentation result
                    masks = segmentation_result[0].masks.data.cpu().numpy() if segmentation_result[0].masks else None
                    detection_boxes = segmentation_result[0].boxes.xyxy.cpu().numpy() if segmentation_result[0].boxes else None

                    if masks is not None and detection_boxes is not None:
                        combined_results = []
                        conversion_factor = 3.5  # 0.35 cm/px * 10 mm/cm = 3.5 mm/px

                        # Draw annotations on the frame
                        annotated_frame = np.array(frame).copy()

                        for idx, (mask, box) in enumerate(zip(masks, detection_boxes)):
                            mask = (mask * 255).astype(np.uint8)  # Convert mask to binary image
                            length_mm, width_mm, area_mm = calculate_crack_properties_approach2(mask, conversion_factor)
                            severity = calculate_severity(width_mm)

                            # Get the class name from the segmentation model's names dictionary
                            class_id = segmentation_result[0].boxes.cls[idx].item()  
                            if class_id in segmentation_result[0].names:  
                                class_name = segmentation_result[0].names[class_id]
                            else:
                                class_name = f"Unknown Class {class_id}"  

                            # Calculate the centroid of the bounding box for placing the identification code
                            x0, y0, x1, y1 = box
                            centroid = (int((x0 + x1) / 2), int((y0 + y1) / 2))

                            # Draw the identification code (A, B, C, etc.) on the frame
                            cv2.putText(
                                annotated_frame,
                                f"{chr(65 + idx)}",  
                                centroid,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0,  
                                (0, 0, 0),  
                                3,  
                                lineType=cv2.LINE_AA,
                            )

                            combined_results.append({
                                'S.No.': idx + 1,  # Serial number starting from 1
                                'Type of Crack': class_name,
                                'Identification Code': chr(65 + idx),
                                'Length (mm)': length_mm,
                                'Width (mm)': width_mm,
                                'Area (mm²)': area_mm,
                                'Severity': severity
                            })

                        # Display the annotated frame
                        st.image(annotated_frame, caption="Frame with Identification Codes", use_container_width=True)

                        # Display results in a table
                        st.write("### Crack Properties")
                        df = pd.DataFrame(combined_results)
                        st.table(df.set_index('S.No.')) 
                    #else:
                        #st.error("No cracks detected in this frame.")
        else:
            st.warning("Please run the identification module first to get the results.")

# Main app
if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Pavement", "Pavement"])
    
    if page == "Pavement":
        pavement_analysis("Pavement")
    else:
        pavement_analysis("Pavement")
