import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading WM-811K dataset...")
df = pd.read_pickle('LSWMD.pkl')
print(f"Total wafers loaded: {len(df)}")

#  most difficult part: "failurtype" column kc clean krna hai as it has different formats data such as string,array ye sb
def clean_label(x):
    # numpy array ko pehle flatten karo
    if isinstance(x, np.ndarray):
        x = x.flatten()
        return x[0] if x.size > 0 and x[0]!= '' else 'none'

    # list case
    elif isinstance(x, list):
        if len(x) == 0:
            return 'none'
        elif isinstance(x[0], list):
            return x[0][0] if len(x[0]) > 0 else 'none'
        elif isinstance(x[0], str):
            return x[0] if x[0]!= '' else 'none'
        else:
            return 'none'

    # string case
    elif isinstance(x, str):
        return x if x!= '' else 'none'

    # NaN case
    elif pd.isna(x):
        return 'none'
    else:
        return 'none'

df['failureType'] = df['failureType'].apply(clean_label)

print("\n=== Defect distribution ===")
print(df['failureType'].value_counts())

# Cleaning
df = df.dropna(subset=['waferMap'])
df = df[df['waferMap'].apply(lambda x: len(x) > 0)]

# Ab 'none' ko hatao, sirf defects rakho
defect_df = df[df['failureType']!= 'none'].copy()
print(f"\nAfter cleaning: {len(defect_df)} defect wafers")
print(f"Total 'none' wafers: {len(df) - len(defect_df)}")

if len(defect_df) == 0:
    print("ERROR: NO DEFECTS FOUND.")
    exit()

# ===== Wafer map plot =====
def plot_sample_wafers():
    plt.figure(figsize=(12,12))
    sample = defect_df.sample(n=min(9, len(defect_df)), random_state=42)

    for i, (_, row) in enumerate(sample.iterrows()):
        plt.subplot(3, 3, i+1)
        plt.imshow(row['waferMap'], cmap='jet')
        plt.title(f"{row['failureType']}", fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('wafer_samples.png')
    plt.show()
    print("Saved: wafer_samples.png")

# ===== Bar chart =====
def plot_defect_dist():
    plt.figure(figsize=(12,6))
    # 'none' ko hata ke plot kar
    plot_data = df[df['failureType']!= 'none']
    order = plot_data['failureType'].value_counts().index
    sns.countplot(data=plot_data, x='failureType', order=order)
    plt.xticks(rotation=45)
    plt.title('Defect Type Distribution - WM811K')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('defect_distribution.png')
    plt.show()
    print("Saved: defect_distribution.png")

print("\nPlotting 9 defect wafers...")
plot_sample_wafers()

print("\nPlotting defect distribution...")
plot_defect_dist()

print("\nEDA Complete ✅")