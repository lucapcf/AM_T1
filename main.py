import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = kagglehub.dataset_download("shashwatwork/dementia-prediction-dataset")
data_path = f"{path}/dementia_dataset.csv"
df = pd.read_csv(data_path)

print("First few rows of the dataset:")
print(df.head())
print("\nStatistical summary:")
print(df.describe())
print("\nMissing data information:")
print(df.isnull().sum())

# Plotting distribution of dementia groups
group_counts = df['Group'].value_counts(normalize=True) * 100
group_counts.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'])
plt.title("Dementia Group Distribution (Percentage)")
plt.xlabel("Group")
plt.ylabel("Percentage (%)")
plt.show()

# Restricting features
age_educ_df = df[['Age', 'EDUC', 'Group']].dropna()
age_educ_df = age_educ_df[age_educ_df['Group'].isin(['Nondemented', 'Demented'])]

y = age_educ_df['Group'].map({'Nondemented': 0, 'Demented': 1})
X = age_educ_df[['Age', 'EDUC']]

# Plotting a violin plot to visualize education years distribution by dementia group
plt.figure(figsize=(10, 7))
sns.violinplot(x=y, y='EDUC', data=age_educ_df, palette='coolwarm', inner='quartile')
plt.title('Violin Plot of Education Years by Dementia Group')
plt.xlabel('Group (0: Nondemented, 1: Demented)')
plt.ylabel('Education years')
plt.xticks(ticks=[0, 1], labels=['Nondemented', 'Demented'])
plt.show()
