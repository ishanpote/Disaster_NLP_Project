import matplotlib.pyplot as plt
import seaborn as sns
import os

# Your exact test dataset distribution from the terminal outputs
categories = ['Low Urgency', 'Medium Urgency', 'High Urgency']
counts = [4032, 1250, 1230]
total = sum(counts)

# Academic styling
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Create the bar chart (using colors that intuitively match urgency)
colors = ['#2ca02c', '#ff7f0e', '#d62728'] # Green, Orange, Red
bars = plt.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.2)

# Add exact numbers and percentages on top of the bars
for bar in bars:
    yval = bar.get_height()
    percentage = f"{(yval / total) * 100:.1f}%"
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, 
             f"{yval}\n({percentage})", ha='center', va='bottom', 
             fontsize=12, fontweight='bold')

# Professional labeling
plt.title("Test Dataset Class Distribution (HumAID)", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Number of Tweets", fontsize=14, fontweight='bold')
plt.xlabel("Urgency Level", fontsize=14, fontweight='bold')
plt.ylim(0, 5000) # Give it some headroom

# Save as a high-resolution PNG for your Word document
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset_Distribution.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Success! Graph saved to: {output_path}")