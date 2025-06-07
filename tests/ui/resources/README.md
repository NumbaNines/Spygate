# Test Resources

This directory contains resources needed for UI component testing.

## Sample Video

The `sample.mp4` file should be a short (5-10 seconds) video file used for testing the VideoTimeline component. You can generate one using FFmpeg:

```bash
# Generate a test video with a black background and white text
ffmpeg -f lavfi -i color=c=black:s=1280x720:d=5 -vf "drawtext=text='Test Video':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2" -c:v libx264 sample.mp4
```

This will create a 5-second video with "Test Video" text centered on a black background. The video is sufficient for testing playback controls, timeline functionality, and frame navigation.
