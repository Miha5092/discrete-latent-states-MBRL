import pandas as pd
import json

# Define a function to load the data into a pandas DataFrame
def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Replace single quotes with double quotes for valid JSON
            corrected_line = line.replace("'", '"')
            try:
                # Convert the JSON string to a dictionary
                data_dict = json.loads(corrected_line)
                data.append(data_dict)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    return df

# Example usage
filename = 'simplified.txt'  # Replace with your actual file path
df = load_data(filename)

df = df[(df["exploration_constant"] == 1.0) &  (df["planning_horizon"] == 32) & (df["num_simulations"] == 128) & (df["discount_factor"] == 0.97)]

# now for each hyperparameter, we plot the mean_reward for each value of this hyperparameter while averaging out the other hyperparameters
import matplotlib.pyplot as plt
import seaborn as sns

for hyperparameter in ['exploration_constant', 'discount_factor', 'planning_horizon', "num_simulations"]:
    # Create a bar plot of mean_reward by hyperparameter value
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=hyperparameter, y='mean_return')
    plt.title(f'Mean Reward by {hyperparameter}')
    plt.show()

# print the top n combination among all combinations and their performance
n = 5
top_n_combinations = df.nlargest(n, 'mean_return')

print(f"Top {n} combinations:")
print(top_n_combinations)