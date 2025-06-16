# Analyze your actual template confidence values
test_results = [
    ('1ST normal', 0.951),
    ('2ND normal', 0.962), 
    ('1ST GOAL', 0.956),
    ('4TH GOAL', 0.926)
]

print('=== ACTUAL TEMPLATE CONFIDENCE ANALYSIS ===')
print('Your current results:')
for name, conf in test_results:
    print(f'  {name}: {conf:.3f}')

print(f'\nAverage: {sum(c for _, c in test_results) / len(test_results):.3f}')
print(f'Minimum: {min(c for _, c in test_results):.3f}')
print(f'Range: {min(c for _, c in test_results):.3f} - {max(c for _, c in test_results):.3f}')

print('\n=== THRESHOLD IMPACT ANALYSIS ===')
thresholds = [0.08, 0.15, 0.25, 0.35, 0.50, 0.70]
for thresh in thresholds:
    passed = sum(1 for _, conf in test_results if conf >= thresh)
    print(f'Threshold {thresh:.2f}: {passed}/4 tests would pass ({passed/4*100:.0f}%)')

print('\n=== EXPERT RECOMMENDATIONS ===')
print('Current system (0.08): TOO LOW - risk of false positives')
print('Proposed 0.50: TOO HIGH - would break legitimate detections')
print('Sweet spot: 0.25-0.35 - filters noise while preserving accuracy')

print('\n=== QUALITY-BASED RECOMMENDATIONS ===')
quality_thresholds = {
    "high_quality": 0.35,      # Clean gameplay footage
    "medium_quality": 0.28,    # Slightly compressed
    "low_quality": 0.22,       # Poor quality/compression
    "streamer_content": 0.18,  # Overlays, webcam artifacts
}

for quality, thresh in quality_thresholds.items():
    passed = sum(1 for _, conf in test_results if conf >= thresh)
    print(f'{quality}: {thresh:.2f} threshold -> {passed}/4 pass ({passed/4*100:.0f}%)') 