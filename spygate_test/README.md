# SpygateAI Test Dataset Organization and Training

This application provides a GUI interface for organizing and training a YOLO model on gameplay footage screenshots from Madden NFL 25 and College Football 25.

## Prerequisites

- Python 3.9 or higher
- Required packages (install via `pip install -r requirements.txt`):
  - ultralytics>=8.1.0
  - opencv-python>=4.8.0
  - PyQt6>=6.6.0
  - torch>=2.1.0
  - PyYAML>=6.0.1

## Directory Structure

Before running the application, ensure you have the following structure:

```
spygate_test/
├── resized_1920x1080/
│   ├── resized_frame_0001.png
│   ├── resized_frame_0001.txt
│   └── ... (other PNG and TXT files)
├── requirements.txt
├── spygate_trainer.py
├── dataset_organizer.py
├── model_trainer.py
└── README.md
```

## Features

1. Dataset Organization:

   - Identifies 50 annotated image-text pairs
   - Moves unannotated images to a separate folder
   - Creates YOLO dataset structure with train/val/test splits (70/20/10)
   - Generates data.yaml configuration file

2. Model Training:

   - Uses YOLOv8n pre-trained model
   - Trains for 50 epochs
   - Uses 640x640 input size
   - Batch size of 8
   - Saves results to runs/test/train_exp

3. GUI Interface:
   - Progress bar for training epochs
   - Real-time log display
   - Error handling and user feedback
   - Clean, modern design

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the GUI application:

   ```bash
   python spygate_trainer.py
   ```

3. Click "Start Training" to begin the process:
   - The application will first organize your dataset
   - Then it will start training the YOLO model
   - Progress and logs will be displayed in real-time

## Output Structure

After running the application, you'll have:

```
spygate_test/
├── resized_1920x1080/
│   └── unannotated/
├── test_dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
└── runs/
    └── test/
        └── train_exp/
            └── (training results)
```

## Error Handling

The application includes comprehensive error handling for:

- Missing files or directories
- Invalid dataset structure
- Training interruptions
- File operation errors

All errors are displayed in the GUI with detailed messages to help troubleshoot issues.

## Support

This script is part of the SpygateAI project for analyzing gameplay footage. For issues or questions, please refer to the project documentation.
