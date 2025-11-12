import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Try to set Korean font
# In the execution environment, 'NanumGothic' is often available.
try:
    plt.rcParams['font.family'] = 'NanumGothic'
    # Test if font is found
    fm.findfont(fm.FontProperties(family='NanumGothic'))
except:
    # Fallback if NanumGothic is not found
    print("NanumGothic font not found, falling back to default sans-serif. Korean text may not display correctly.")
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False # For displaying minus sign correctly

# --- Data ---
models_accuracy = ['EfficientDet', 'YOLOv5']
accuracy_values = [86.7, 99.5]

models_speed = ['EfficientDet', 'YOLOv5']
speed_values = [17.6, 30]

# Colors from the user's last accepted image (Green for EfficientDet, Blue for YOLOv5)
colors = ['#6aa84f', '#4a86e8'] # Matching the approximate green/blue from the image

# Reverse the order for plotting so YOLOv5 is on top, as in the user's initial request
models_accuracy.reverse()
accuracy_values.reverse()
models_speed.reverse()
speed_values.reverse()
colors.reverse() # Apply reverse to colors as well

y_pos_accuracy = np.arange(len(models_accuracy))
y_pos_speed = np.arange(len(models_speed))


# --- Create Figure and Axes ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig.patch.set_facecolor('#f5f5f5') # Light gray background

# --- Plot 1: Accuracy Comparison (정확도 비교) ---
ax1.barh(y_pos_accuracy, accuracy_values, color=colors, height=0.6)
ax1.set_title('📊 정확도 비교', fontsize=16, fontweight='bold', pad=20, color='#333')
ax1.set_facecolor('#f5f5f5')

# Set Y-ticks
ax1.set_yticks(y_pos_accuracy)
ax1.set_yticklabels(models_accuracy, fontsize=14, color='#333')

# Remove X-ticks
ax1.set_xticks([])

# Remove spines (borders)
for spine in ['top', 'right', 'bottom', 'left']:
    ax1.spines[spine].set_visible(False)

# Set X-limit to give space for labels
ax1.set_xlim(0, 110)

# Add text labels
ax1.text(accuracy_values[0] + 1, y_pos_accuracy[0], f'{accuracy_values[0]}%', va='center', ha='left', fontsize=14, fontweight='bold', color='#333')
ax1.text(accuracy_values[1] + 1, y_pos_accuracy[1], f'{accuracy_values[1]}%', va='center', ha='left', fontsize=14, fontweight='bold', color='#333')

# --- Plot 2: Inference Speed Comparison (추론 속도 비교) ---
ax2.barh(y_pos_speed, speed_values, color=colors, height=0.6)
ax2.set_title('⚡ 추론 속도 비교', fontsize=16, fontweight='bold', pad=20, color='#333')
ax2.set_facecolor('#f5f5f5')

# Set Y-ticks
ax2.set_yticks(y_pos_speed)
ax2.set_yticklabels(models_speed, fontsize=14, color='#333')

# Remove X-ticks
ax2.set_xticks([])

# Remove spines
for spine in ['top', 'right', 'bottom', 'left']:
    ax2.spines[spine].set_visible(False)

# Set X-limit
ax2.set_xlim(0, 35)

# Add text labels
ax2.text(speed_values[0] + 0.5, y_pos_speed[0], f'{speed_values[0]} FPS', va='center', ha='left', fontsize=14, fontweight='bold', color='#333')
ax2.text(speed_values[1] + 0.5, y_pos_speed[1], f'{speed_values[1]} FPS', va='center', ha='left', fontsize=14, fontweight='bold', color='#333')

# --- Final Layout and Save ---
plt.tight_layout(h_pad=4) # Add vertical padding between plots
plt.savefig('model_comparison_charts_final.png', dpi=150, facecolor=fig.get_facecolor())

print("Chart saved as 'model_comparison_charts_final.png'")