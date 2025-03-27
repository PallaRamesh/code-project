import matplotlib.pyplot as plt
import seaborn as sns

# Sample data for Gender distribution
gender_data = ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Other', 'Female', 'Male']

# Create a count plot (bar chart)
sns.countplot(x=gender_data, palette='Set2')

# Add title and labels
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')

# Show the plot
plt.show()
