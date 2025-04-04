from ultralytics import YOLO
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

class YOLOFeatureVisualizer:
    """
    A class to visualize intermediate features from YOLO models.
    This captures and visualizes features after the backbone and after the neck.
    """
    def __init__(self, model_path='yolo11n.pt'):
        """Initialize with the specified YOLO model"""
        self.model = YOLO(model_path)
        self.model.model.eval()  # Set to evaluation mode
        
        # Identify model boundaries
        self.backbone_end_idx, self.neck_end_idx = self._find_model_boundaries()
        print(f"Model structure identified: backbone ends at layer {self.backbone_end_idx}, neck ends at layer {self.neck_end_idx}")
    
    def _find_model_boundaries(self):
        """Find the indices where backbone ends and neck ends"""
        backbone_end_idx = None
        neck_end_idx = None
        
        # Get the PyTorch model
        model = self.model.model
        
        # Debug model structure
        print("Model structure:")
        for i, m in enumerate(model.model):
            print(f"Layer {i}: {type(m).__name__}")
        
        # Find SPPF/SPP layer (end of backbone)
        for i, m in enumerate(model.model):
            layer_type = type(m).__name__
            if layer_type in ['SPPF', 'SPP'] and backbone_end_idx is None:
                backbone_end_idx = i
                print(f"Found backbone end at layer {i} ({layer_type})")
        
        # Find Detect layer or similar (start of head)
        detect_idx = None
        for i, m in enumerate(model.model):
            if any(x in type(m).__name__.lower() for x in ['detect', 'head', 'rtdetr']) or hasattr(m, 'anchors'):
                detect_idx = i
                print(f"Found detection head at layer {i} ({type(m).__name__})")
                break
        
        # The neck ends right before the detect layer
        if detect_idx:
            neck_end_idx = detect_idx - 1
        
        # Fallbacks if heuristics fail
        if backbone_end_idx is None:
            backbone_end_idx = len(model.model) // 3
            print(f"Using fallback backbone end: layer {backbone_end_idx}")
            
        if neck_end_idx is None:
            neck_end_idx = len(model.model) - 2
            print(f"Using fallback neck end: layer {neck_end_idx}")
        
        return backbone_end_idx, neck_end_idx
    
    def process_image(self, img_path):
        """
        Process an image through the model and capture intermediate outputs
        
        Args:
            img_path: Path to the image file
        
        Returns:
            tuple: (results, backbone_output, neck_output)
        """
        # Load image
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_path
        
        # Keep a copy of original image for visualization
        original_img = img.copy()
        
        # Get the results using standard prediction
        results = self.model(img)
        
        # Capture intermediate outputs
        backbone_output, neck_output = self._get_intermediate_outputs(img)
        
        return results, backbone_output, neck_output, original_img
    
    def _get_intermediate_outputs(self, img):
        """
        Run the model to get intermediate outputs
        
        Args:
            img: Image to process
        
        Returns:
            tuple: (backbone_output, neck_output)
        """
        # Preprocess image
        input_tensor = self._preprocess_image(img)
        
        # Get model
        model = self.model.model
        
        # Run model with captures
        with torch.no_grad():
            try:
                backbone_output, neck_output = self._run_with_captures(model, input_tensor)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                # Try alternative approach
                print("Trying alternative approach...")
                backbone_output, neck_output = self._alternative_feature_extraction(model, input_tensor)
        
        return backbone_output, neck_output
    
    def _alternative_feature_extraction(self, model, x):
        """Alternative method to extract features using hooks"""
        backbone_features = []
        neck_features = []
        
        # Define hook functions
        def backbone_hook(module, input, output):
            backbone_features.append(output.clone())
            
        def neck_hook(module, input, output):
            neck_features.append(output.clone())
        
        # Register hooks
        backbone_module = model.model[self.backbone_end_idx]
        neck_module = model.model[self.neck_end_idx]
        
        handle1 = backbone_module.register_forward_hook(backbone_hook)
        handle2 = neck_module.register_forward_hook(neck_hook)
        
        # Forward pass
        model(x)
        
        # Remove hooks
        handle1.remove()
        handle2.remove()
        
        # Return captured features
        return backbone_features[0], neck_features[0]
    
    def _preprocess_image(self, img):
        """Preprocess image for model input"""
        # Manual preprocessing similar to YOLO's internal preprocessing
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img)
        
        # Get model image size
        if hasattr(self.model, 'args') and hasattr(self.model.args, 'imgsz'):
            imgsz = self.model.args.imgsz
        else:
            imgsz = 640  # default YOLO size
        
        # Resize and convert to tensor
        img_tensor = self._resize_and_to_tensor(img_pil, imgsz)
        
        return img_tensor
        
    def _resize_and_to_tensor(self, img, imgsz):
        """Resize image and convert to tensor"""
        # Resize
        r = imgsz / max(img.size)
        if r != 1:
            img = img.resize((int(img.width * r), int(img.height * r)), Image.BILINEAR)
            
        # Convert to tensor
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        
        # Normalize
        img = img / 255.0
        
        # Add batch dimension
        if img.ndim == 3:
            img = img.unsqueeze(0)
            
        # Move to same device as model
        device = next(self.model.model.parameters()).device
        img = img.to(device)
        
        return img
    
    def _run_with_captures(self, model, x):
        """Run model forward pass capturing intermediate outputs"""
        backbone_output = None
        neck_output = None
        
        # Run model forward pass (mimicking BaseModel._predict_once)
        y = []  # outputs
        for i, m in enumerate(model.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            x = m(x)  # run
            
            # Save output if needed for later layers
            if hasattr(model, 'save') and i in model.save:
                y.append(x)
            else:
                y.append(None)
            
            # Capture outputs at boundaries
            if i == self.backbone_end_idx:
                backbone_output = x.clone() if not isinstance(x, list) else [t.clone() for t in x]
            elif i == self.neck_end_idx:
                neck_output = x.clone() if not isinstance(x, list) else [t.clone() for t in x]
        
        return backbone_output, neck_output
    
    def visualize_features(self, backbone_output, neck_output, max_channels=16):
        """
        Visualize backbone and neck feature maps
        
        Args:
            backbone_output: Output tensor from backbone
            neck_output: Output tensor from neck
            max_channels: Maximum number of channels to visualize
        """
        # Create a figure with two rows
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Visualize backbone features
        self._visualize_feature_grid(backbone_output, "Backbone Output", axes[0], max_channels)
        
        # Visualize neck features
        self._visualize_feature_grid(neck_output, "Neck Output", axes[1], max_channels)
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_feature_grid(self, features, title, ax, max_channels=16):
        """
        Visualize feature maps in a grid within the provided axis
        
        Args:
            features: Feature tensor
            title: Title for the plot
            ax: Matplotlib axis
            max_channels: Maximum number of channels to visualize
        """
        # Handle different input types
        if isinstance(features, list):
            # If it's a list of tensors (like from neck), use the first one
            features = features[0] if features else None
        
        if features is None:
            ax.text(0.5, 0.5, f"No {title} features available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return
        
        # Convert to numpy for visualization
        features = features.detach().cpu().numpy()
        
        # Get first item if it's a batch
        if features.ndim > 3:
            features = features[0]
        
        # For single channel features, expand dimensions
        if features.ndim == 2:
            features = features[np.newaxis, :, :]
        
        # Calculate mean feature map for background
        mean_feature = np.mean(features, axis=0)
        vmin, vmax = mean_feature.min(), mean_feature.max()
        if vmax > vmin:
            mean_feature = (mean_feature - vmin) / (vmax - vmin)
        
        # Display mean feature as background
        ax.imshow(mean_feature, cmap='gray', alpha=0.5)
        ax.set_title(f"{title} - {features.shape[0]} channels", fontsize=14)
        
        # Limit number of channels to display
        channels = min(features.shape[0], max_channels)
        
        # Create a grid of smaller subplots for individual channels
        n_cols = 4
        n_rows = (channels + n_cols - 1) // n_cols
        
        # Calculate grid positions
        h, w = features.shape[1], features.shape[2]
        cell_h, cell_w = h // n_rows, w // n_cols
        
        # Plot individual channels as small overlays
        for i in range(channels):
            row, col = i // n_cols, i % n_cols
            
            # Extract and normalize feature map
            feature = features[i]
            vmin, vmax = feature.min(), feature.max()
            if vmax > vmin:
                feature = (feature - vmin) / (vmax - vmin)
            
            # Calculate position in grid
            y, x = row * cell_h, col * cell_w
            
            # Create inset axes for this channel
            inset_ax = ax.inset_axes([x/w, y/h, cell_w/w, cell_h/h])
            
            # Plot feature map
            inset_ax.imshow(feature, cmap='viridis')
            inset_ax.set_title(f"Ch {i}", fontsize=8)
            inset_ax.axis('off')
    
    def visualize_detection_results(self, img, results, figsize=(10, 10)):
        """
        Visualize detection results on the input image
        
        Args:
            img: Input image
            results: YOLO detection results
            figsize: Figure size
        """
        # Plot the image
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Detection Results", fontsize=14)
        
        # Draw boxes for each detection
        for r in results:
            for box in r.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get class and confidence
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Get class name
                cls_name = self.model.names[cls_id]
                
                # Create label
                label = f"{cls_name} {conf:.2f}"
                
                # Draw box and label
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                plt.gca().text(x1, y1-5, label, 
                             bbox=dict(facecolor='red', alpha=0.5),
                             fontsize=10, color='white')
        
        plt.tight_layout()
        plt.show()
    
    def process_and_visualize(self, img_path, max_channels=16):
        """
        Process an image and visualize both detections and intermediate features
        
        Args:
            img_path: Path to the image
            max_channels: Maximum number of channels to visualize
        """
        # Process image
        results, backbone_output, neck_output, original_img = self.process_image(img_path)
        
        # Visualize detection results
        self.visualize_detection_results(original_img, results)
        
        # Visualize intermediate features
        self.visualize_features(backbone_output, neck_output, max_channels)
        
        return results, backbone_output, neck_output

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Model path')
    parser.add_argument('--img', type=str, required=True, default="bus.jpg", help='Image path')
    parser.add_argument('--max-channels', type=int, default=16, help='Maximum number of channels to visualize')
    args = parser.parse_args()
    
    # Initialize visualizer with model
    print(f"Loading model: {args.model}")
    visualizer = YOLOFeatureVisualizer(args.model)
    
    # Process and visualize an image
    print(f"Processing image: {args.img}")
    results, backbone_features, neck_features = visualizer.process_and_visualize(args.img, args.max_channels)
    
    # Print feature shapes if available
    if backbone_features is not None:
        if isinstance(backbone_features, list):
            print(f"Backbone features: {len(backbone_features)} tensors")
            for i, feat in enumerate(backbone_features):
                print(f"  Backbone tensor {i} shape: {feat.shape}")
        else:
            print(f"Backbone feature shape: {backbone_features.shape}")
    
    if neck_features is not None:
        if isinstance(neck_features, list):
            print(f"Neck features: {len(neck_features)} tensors")
            for i, feat in enumerate(neck_features):
                print(f"  Neck tensor {i} shape: {feat.shape}")
        else:
            print(f"Neck feature shape: {neck_features.shape}")