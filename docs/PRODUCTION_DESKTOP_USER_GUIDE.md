# SpygateAI Production Desktop Application

## User Guide

**Version:** 1.0.0
**Application:** `spygate_production_desktop.py`
**Interface Style:** FACEIT-inspired Professional Gaming UI

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Application Interface](#application-interface)
3. [Video Import Workflow](#video-import-workflow)
4. [Analysis Pipeline](#analysis-pipeline)
5. [Clip Review and Management](#clip-review-and-management)
6. [Export Functionality](#export-functionality)
7. [Settings and Configuration](#settings-and-configuration)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)

---

## Getting Started

### System Requirements

**Minimum Hardware:**

- **GPU:** DirectX 11 compatible graphics card
- **RAM:** 8GB system memory
- **Storage:** 2GB free space for application and temporary files
- **OS:** Windows 10/11, macOS 10.14+, Linux Ubuntu 18.04+

**Recommended Hardware:**

- **GPU:** NVIDIA RTX series with 8GB+ VRAM (for ULTRA tier performance)
- **RAM:** 16GB+ system memory
- **Storage:** SSD with 10GB+ free space
- **CPU:** Multi-core processor (8+ cores recommended)

### Installation and Launch

1. **Ensure Python Environment:** Make sure you have Python 3.8+ installed with all dependencies
2. **Launch Application:** Run the command:
   ```bash
   python spygate_production_desktop.py
   ```
3. **Wait for Initialization:** The app will detect your hardware and configure optimizations automatically

### Hardware Tier Detection

The application automatically detects your hardware capabilities:

- **ULTRA Tier:** RTX 4070/4080/4090 with 12GB+ VRAM (15-frame analysis intervals)
- **HIGH Tier:** RTX 3070/3080 or equivalent with 8GB+ VRAM (30-frame intervals)
- **MEDIUM Tier:** GTX 1660/RTX 2060 or equivalent with 4GB+ VRAM (60-frame intervals)
- **LOW Tier:** Integrated graphics or older cards (90-frame intervals)

_Your tier is displayed in the sidebar and affects processing speed and quality._

---

## Application Interface

### FACEIT-Style Design Elements

The application features a professional gaming interface inspired by FACEIT:

- **Dark Theme:** Deep blacks (#0f0f0f) and charcoal grays (#1a1a1a)
- **Orange Accents:** Professional orange highlights (#ff6b35) for interactive elements
- **Responsive Layout:** Adapts to different screen sizes and resolutions
- **Professional Typography:** Clean, readable fonts optimized for gaming applications

### Main Layout Components

#### 1. Sidebar Navigation (250px)

**Location:** Left side of the application
**Sections:**

- **Analysis:** Video import, processing controls, and progress tracking
- **Review:** Clip management, approval workflow, and export options
- **Settings:** Application configuration, hardware optimization, and preferences

#### 2. Main Content Area

**Dynamic Content Based on Selected Panel:**

- **Drop Zone:** Large, prominent area for video file import
- **Progress Display:** Real-time analysis progress with detailed metrics
- **Clip Grid:** 4-clips-per-row layout for reviewing detected highlights
- **Settings Interface:** Configuration options and system information

#### 3. Status Bar

**Location:** Bottom of the application
**Information Displayed:**

- Hardware tier status (ULTRA/HIGH/MEDIUM/LOW)
- Current operation status
- Processing statistics and performance metrics

---

## Video Import Workflow

### Supported Video Formats

The application supports multiple professional video formats:

- **MP4:** Most common format, recommended for best compatibility
- **MOV:** Apple/professional video format
- **AVI:** Legacy format with broad compatibility
- **MKV:** High-quality container format

### Import Methods

#### Method 1: Drag and Drop (Recommended)

1. **Navigate to Analysis Panel:** Click "Analysis" in the sidebar
2. **Locate Drop Zone:** Large central area with dashed border
3. **Drag Video File:** Drag your video file from file explorer
4. **Drop on Zone:** Release the file over the highlighted drop zone
5. **Automatic Validation:** Application validates format and prepares for analysis

#### Method 2: File Dialog (Alternative)

1. **Click Browse Button:** Located within the drop zone
2. **Select Video File:** Use the file dialog to navigate and select your video
3. **Confirm Selection:** Click "Open" to import the selected video

### Visual Feedback During Import

- **Hover State:** Drop zone highlights when a valid file is dragged over
- **Validation:** Green border indicates a valid video file
- **Error State:** Red border with error message for invalid files
- **Loading:** Progress indicator shows file loading status

---

## Analysis Pipeline

### Automatic Processing

Once a video is imported, the analysis pipeline begins automatically:

#### Stage 1: Video Preprocessing

- **Duration:** 2-5 seconds for most videos
- **Process:** Video format validation, metadata extraction, frame counting
- **Display:** Basic video information and estimated processing time

#### Stage 2: Hardware-Adaptive Analysis

**Processing varies by hardware tier:**

- **ULTRA Tier (RTX 4070+):** Analyzes every 15th frame for maximum accuracy
- **HIGH Tier (RTX 3070+):** Analyzes every 30th frame for balanced performance
- **MEDIUM Tier (GTX 1660+):** Analyzes every 60th frame for stable performance
- **LOW Tier (Integrated):** Analyzes every 90th frame for basic functionality

#### Stage 3: Situation Detection

The AI identifies key football moments:

- **3rd & Long:** Critical down and distance situations
- **Red Zone:** Scoring opportunity situations
- **Turnover:** Interceptions, fumbles, and possession changes
- **Scoring Play:** Touchdowns, field goals, and safety plays
- **Big Play:** Significant yardage gains (20+ yards)

#### Stage 4: Clip Generation

- **Automatic Extraction:** Creates 10-15 second clips around detected moments
- **Context Preservation:** Includes pre/post-play action for complete context
- **Quality Optimization:** Maintains original video quality in extracted clips

### Progress Tracking

**Real-Time Updates Include:**

- **Frame Progress:** "Processing frame 1,250 of 36,004"
- **Percentage Complete:** Visual progress bar with percentage
- **Processing Speed:** Frames per second processing rate
- **Estimated Time:** Remaining processing time calculation
- **Detected Clips:** Running count of identified highlight moments

---

## Clip Review and Management

### Grid-Based Interface

#### Layout Structure

- **4 Clips Per Row:** Optimized for modern widescreen displays
- **Thumbnail Previews:** Large, clear preview images for each clip
- **Metadata Display:** Situation type, timestamp, and confidence score
- **Action Buttons:** Approve, reject, and preview controls for each clip

#### Clip Information Display

**Each clip widget shows:**

- **Thumbnail:** Representative frame from the detected moment
- **Situation Type:** Label (e.g., "3rd & Long", "Red Zone Play")
- **Timestamp:** Video position (e.g., "12:34 - 12:49")
- **Confidence Score:** AI detection confidence (e.g., "92% confidence")
- **Duration:** Clip length in seconds

### Approval Workflow

#### Approve Clips

1. **Review Thumbnail:** Check the preview image for relevance
2. **Click Approve:** Green checkmark button on the clip widget
3. **Visual Confirmation:** Clip widget highlights in green
4. **Queue for Export:** Approved clips are added to export queue

#### Reject Clips

1. **Identify Unwanted Clips:** False positives or irrelevant moments
2. **Click Reject:** Red X button on the clip widget
3. **Visual Confirmation:** Clip widget dims and shows rejected status
4. **Remove from Queue:** Rejected clips are excluded from export

#### Preview Clips (Future Feature)

1. **Click Preview Button:** Play button icon on clip widget
2. **Mini Player:** Small video player opens showing the full clip
3. **Playback Controls:** Play, pause, and scrub through the clip
4. **Make Decision:** Approve or reject after previewing

### Bulk Operations

- **Select All:** Checkbox to approve all detected clips at once
- **Select None:** Checkbox to reject all clips and start over
- **Filter by Type:** Show only specific situation types (e.g., only "Big Play")
- **Sort Options:** Arrange clips by timestamp, confidence, or situation type

---

## Export Functionality

### Export Process

#### 1. Review Approved Clips

- **Navigate to Review Panel:** Click "Review" in sidebar
- **Check Approved List:** View all clips marked for export
- **Final Review:** Make any last-minute approval changes

#### 2. Select Export Directory

1. **Click Export Button:** Large orange button in the Review panel
2. **Directory Dialog:** System file browser opens
3. **Choose Location:** Navigate to desired export folder
4. **Confirm Selection:** Click "Select Folder" to confirm location

#### 3. Batch Export Process

- **Automatic Processing:** All approved clips exported simultaneously
- **Progress Tracking:** Real-time progress bar for export status
- **File Naming:** Clips named with timestamps and situation types
- **Quality Preservation:** Original video quality maintained in exported clips

### Export File Structure

**Default Naming Convention:**

```
SpygateAI_Export_YYYY-MM-DD_HHMMSS/
├── BigPlay_001_12m34s.mp4
├── RedZone_002_18m12s.mp4
├── ThirdLong_003_23m45s.mp4
└── Export_Summary.txt
```

**Export Summary Contents:**

- Video source information
- Total clips exported
- Processing statistics
- Hardware tier used
- Export timestamp

---

## Settings and Configuration

### Application Settings

#### Performance Options

- **Hardware Tier Override:** Manually set tier if auto-detection is incorrect
- **Frame Skip Adjustment:** Fine-tune analysis intervals for your hardware
- **Memory Usage Limit:** Control maximum RAM usage during processing
- **CPU Core Utilization:** Set number of CPU cores for processing

#### Analysis Settings

- **Detection Sensitivity:** Adjust AI confidence thresholds
- **Situation Types:** Enable/disable specific situation detection
- **Clip Duration:** Customize length of extracted clips (5-30 seconds)
- **Quality Settings:** Balance processing speed vs. analysis accuracy

#### Export Settings

- **Default Export Location:** Set preferred folder for exported clips
- **File Naming Convention:** Customize clip naming patterns
- **Video Quality:** Choose export quality (Original, High, Medium)
- **Metadata Inclusion:** Include/exclude technical metadata in clips

#### UI Preferences

- **Theme Options:** Adjust color scheme and contrast
- **Panel Layout:** Customize sidebar width and panel arrangement
- **Progress Display:** Choose detailed vs. simplified progress information
- **Notification Settings:** Configure completion alerts and sounds

---

## Troubleshooting

### Common Issues and Solutions

#### Application Won't Launch

**Symptoms:** Error messages on startup, crashes during initialization
**Solutions:**

1. Check Python environment and dependencies
2. Update GPU drivers to latest version
3. Verify sufficient available RAM and storage
4. Run application as administrator (Windows)

#### Hardware Detection Issues

**Symptoms:** Incorrect tier detection, poor performance
**Solutions:**

1. Update GPU drivers
2. Close other GPU-intensive applications
3. Manually override hardware tier in settings
4. Check for Windows GPU scheduler conflicts

#### Video Import Failures

**Symptoms:** Files rejected, import errors, format not supported
**Solutions:**

1. Verify video file integrity (not corrupted)
2. Convert to MP4 format using media converter
3. Check file size (avoid extremely large files >4GB)
4. Ensure sufficient storage space for processing

#### Slow Processing Performance

**Symptoms:** Very slow analysis, high memory usage, system freezing
**Solutions:**

1. Close unnecessary background applications
2. Lower detection sensitivity in settings
3. Increase frame skip intervals
4. Process shorter video segments
5. Check thermal throttling on GPU/CPU

#### Export Problems

**Symptoms:** Export failures, missing clips, permission errors
**Solutions:**

1. Ensure write permissions to export directory
2. Check available storage space
3. Close other applications accessing export folder
4. Try different export location
5. Verify approved clips are properly selected

---

## Performance Optimization

### Hardware-Specific Optimization

#### For ULTRA Tier Users (RTX 4070+)

- **Maximize Quality:** Use lowest frame skip settings (15 frames)
- **Enable All Features:** Turn on all situation detection types
- **High Resolution:** Process at full video resolution
- **Parallel Processing:** Utilize all available GPU compute units

#### For HIGH Tier Users (RTX 3070+)

- **Balanced Settings:** Use moderate frame skip (30 frames)
- **Selective Detection:** Focus on most important situation types
- **Moderate Resolution:** Scale processing resolution as needed
- **Memory Management:** Monitor VRAM usage during processing

#### For MEDIUM Tier Users (GTX 1660+)

- **Performance Focus:** Increase frame skip intervals (60 frames)
- **Essential Detection:** Enable only critical situation types
- **Lower Resolution:** Reduce processing resolution for speed
- **Sequential Processing:** Process videos one at a time

#### For LOW Tier Users (Integrated Graphics)

- **Maximum Efficiency:** Use highest frame skip (90 frames)
- **Minimal Detection:** Select single most important situation type
- **Lowest Resolution:** Minimize processing resolution
- **CPU Optimization:** Rely on CPU processing with GPU acceleration disabled

### General Performance Tips

1. **Close Background Apps:** Free up system resources before processing
2. **Process During Off-Hours:** Better performance when system isn't busy
3. **Use SSD Storage:** Faster file access improves processing speed
4. **Adequate Cooling:** Prevent thermal throttling with proper ventilation
5. **Regular Maintenance:** Keep drivers updated and system clean
6. **Batch Processing:** Process multiple short videos rather than one long video

---

## Advanced Features

### Keyboard Shortcuts

- **Ctrl + O:** Open file dialog for video import
- **Ctrl + E:** Export approved clips
- **Ctrl + A:** Select all clips for approval
- **Ctrl + D:** Deselect all clips
- **Space:** Play/pause preview (when available)
- **F11:** Toggle fullscreen mode
- **Ctrl + ,:** Open settings dialog

### Command Line Options

Launch the application with additional parameters:

```bash
# Force specific hardware tier
python spygate_production_desktop.py --tier=HIGH

# Set custom log level
python spygate_production_desktop.py --log-level=DEBUG

# Specify configuration file
python spygate_production_desktop.py --config=custom_config.json
```

### Integration with Other Tools

- **Professional Video Editors:** Exported clips work with Premiere Pro, DaVinci Resolve
- **Team Analysis:** Share export folders with coaches and teammates
- **Cloud Storage:** Upload exports to Google Drive, Dropbox for remote access
- **Streaming Software:** Use clips in OBS, XSplit for content creation

---

## Support and Resources

### Getting Help

1. **Check This Documentation:** Most questions answered in this guide
2. **Review Error Messages:** Application provides detailed error information
3. **Check System Requirements:** Ensure your hardware meets minimum specs
4. **Update Dependencies:** Keep Python packages and drivers current

### Performance Monitoring

The application provides detailed performance metrics:

- **Processing Speed:** Frames per second analysis rate
- **Memory Usage:** RAM and VRAM consumption tracking
- **Hardware Utilization:** GPU and CPU usage percentages
- **Thermal Status:** Temperature monitoring and throttling detection

### Best Practices

1. **Regular Updates:** Keep application and dependencies updated
2. **Hardware Maintenance:** Clean systems and update drivers regularly
3. **Workflow Optimization:** Develop efficient video processing routines
4. **Backup Important Clips:** Save critical exports to multiple locations
5. **Performance Monitoring:** Track system performance for optimization opportunities

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-09
**Application Version:** Production Release
**Compatibility:** Windows 10/11, macOS 10.14+, Linux Ubuntu 18.04+
