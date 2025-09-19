import React from 'react';
import { CheatSheetSection } from '../types';
import CodeBlock from '../components/CodeBlock';

const PythonFundamentalsCheatSheet = () => (
  <div>
    <h3 className="text-xl font-semibold mb-3 text-sky-600 dark:text-sky-400">Core Definitions</h3>
    <ul className="list-disc list-inside space-y-3 mb-6">
      <li><strong>Dynamic Typing:</strong> You don't need to declare a variable's type. Python determines it at runtime based on the assigned value. A variable can hold an integer, then a string.</li>
      <li><strong>Mutability:</strong> An object is mutable if its state or contents can be changed after creation. Lists, Dictionaries, and Sets are mutable.</li>
      <li><strong>Immutability:</strong> An object is immutable if it cannot be changed after creation. Strings, Tuples, Integers, and Floats are immutable. Operations that seem to modify them actually create a new object.</li>
    </ul>

    <h3 className="text-xl font-semibold mb-3 text-sky-600 dark:text-sky-400">Data Structures Comparison (High-Yield Exam Topic)</h3>
    <div className="overflow-x-auto mb-6 not-prose">
      <table className="w-full text-left border-collapse text-sm">
        <thead>
          <tr className="bg-slate-100 dark:bg-slate-700/50">
            <th className="p-3 border border-slate-200 dark:border-slate-700">Characteristic</th>
            <th className="p-3 border border-slate-200 dark:border-slate-700">List</th>
            <th className="p-3 border border-slate-200 dark:border-slate-700">Tuple</th>
            <th className="p-3 border border-slate-200 dark:border-slate-700">Set</th>
            <th className="p-3 border border-slate-200 dark:border-slate-700">Dictionary</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-white dark:bg-slate-800">
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-semibold">Syntax</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-mono">[1, 'a']</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-mono">(1, 'a')</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-mono">{'{1, \'a\'}'}</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-mono">{'{'key': 1}'}</td>
          </tr>
          <tr className="bg-white dark:bg-slate-800">
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-semibold">Mutable</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700 text-green-600 dark:text-green-400">Yes</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700 text-red-600 dark:text-red-400">No</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700 text-green-600 dark:text-green-400">Yes</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700 text-green-600 dark:text-green-400">Yes</td>
          </tr>
          <tr className="bg-white dark:bg-slate-800">
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-semibold">Ordered</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Yes</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Yes</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">No</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Yes (since 3.7)</td>
          </tr>
          <tr className="bg-white dark:bg-slate-800">
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-semibold">Duplicates</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Allowed</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Allowed</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Not Allowed</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Keys must be unique</td>
          </tr>
          <tr className="bg-white dark:bg-slate-800">
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-semibold">Access</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">By integer index</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">By integer index</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">No indexing</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">By key</td>
          </tr>
          <tr className="bg-white dark:bg-slate-800">
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-semibold">Performance<br/>(Membership Test)</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Slow (O(n))</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Slow (O(n))</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Very Fast (O(1))</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Very Fast (O(1))</td>
          </tr>
           <tr className="bg-white dark:bg-slate-800">
            <td className="p-3 border border-slate-200 dark:border-slate-700 font-semibold">Use Case</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Ordered collection, may contain duplicates.</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Immutable data, like coordinates or database records.</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Store unique items, fast membership tests.</td>
            <td className="p-3 border border-slate-200 dark:border-slate-700">Store key-value pairs, fast lookups by key.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Exam-Pattern Code Snippets</h3>
    <p className="mb-4">These snippets reflect patterns seen in past papers and cover essential skills.</p>
    
    <h4 className="font-semibold text-md text-slate-700 dark:text-slate-300">Type Casting User Input (PYP 2022)</h4>
    <CodeBlock code={`
# The input() function ALWAYS returns a string.
age_str = input("Enter your age: ") 
# You must cast it to a numeric type for calculations.
try:
    age_int = int(age_str)
    print(f"Next year, you will be {age_int + 1}")
except ValueError:
    print("Invalid input. Please enter a number.")
    `} language="python" />

    <h4 className="font-semibold text-md text-slate-700 dark:text-slate-300 mt-4">Sorting a List of Dictionaries (PYP 2022)</h4>
    <CodeBlock code={`
smartphones = [
    {'name': 'Galaxy S22', 'price': 699},
    {'name': 'iPhone 14', 'price': 999},
    {'name': 'Pixel 6', 'price': 599}
]
# Use a lambda function in the 'key' argument to specify which field to sort by.
# reverse=True for descending order.
sorted_desc = sorted(smartphones, key=lambda phone: phone['price'], reverse=True)
# sorted_desc[0] will be the iPhone 14 dictionary.
    `} language="python" />

    <h4 className="font-semibold text-md text-slate-700 dark:text-slate-300 mt-4">Defining a Function for a Sequence (PYP 2024)</h4>
     <CodeBlock code={`
def generate_fibonacci(n_terms):
    """Generates the Fibonacci sequence up to n terms."""
    if n_terms <= 0: return []
    if n_terms == 1: return [0]
    
    sequence = [0, 1]
    while len(sequence) < n_terms:
        next_term = sequence[-1] + sequence[-2]
        sequence.append(next_term)
    return sequence

print(generate_fibonacci(7)) # Output: [0, 1, 1, 2, 3, 5, 8]
    `} language="python" />
    
    <h4 className="font-semibold text-md text-slate-700 dark:text-slate-300 mt-4">List Comprehensions</h4>
    <CodeBlock code={`
# A concise way to create lists. Syntax: [expression for item in iterable if condition]
squares = [x**2 for x in range(10) if x % 2 == 0]
# Output: [0, 4, 16, 36, 64]
    `} language="python" />
  </div>
);

const DataSciencePipelineCheatSheet = () => (
    <div>
        <p className="mb-4">The Data Science Pipeline is a systematic framework for tackling a data science project from start to finish. Understanding these stages is a common exam topic.</p>
        <dl className="space-y-4">
            <div>
                <dt className="font-bold text-md text-slate-800 dark:text-slate-100">1. Problem Definition & Data Acquisition</dt>
                <dd className="ml-4 text-slate-600 dark:text-slate-300">Translate a business problem into a data question. Gather data from sources like CSVs, SQL databases, or APIs.</dd>
            </div>
            <div>
                <dt className="font-bold text-md text-slate-800 dark:text-slate-100">2. Data Preparation & Cleaning (Wrangling)</dt>
                <dd className="ml-4 text-slate-600 dark:text-slate-300">The most time-consuming step. Involves handling missing values, correcting data types, removing duplicates, and standardizing data.</dd>
            </div>
            <div>
                <dt className="font-bold text-md text-slate-800 dark:text-slate-100">3. Exploratory Data Analysis (EDA)</dt>
                <dd className="ml-4 text-slate-600 dark:text-slate-300">Investigate the data using visualizations (histograms, scatter plots) and summary statistics to find patterns, anomalies, and form hypotheses.</dd>
            </div>
            <div>
                <dt className="font-bold text-md text-slate-800 dark:text-slate-100">4. Modeling & Machine Learning</dt>
                <dd className="ml-4 text-slate-600 dark:text-slate-300">Select an appropriate algorithm (e.g., Regression, Classification), split data into train/test sets, train the model, and evaluate its performance.</dd>
            </div>
            <div>
                <dt className="font-bold text-md text-slate-800 dark:text-slate-100">5. Visualization & Communication</dt>
                <dd className="ml-4 text-slate-600 dark:text-slate-300">Create clear, explanatory visualizations and communicate findings to stakeholders in a compelling "data story".</dd>
            </div>
             <div>
                <dt className="font-bold text-md text-slate-800 dark:text-slate-100">6. Deployment & Monitoring</dt>
                <dd className="ml-4 text-slate-600 dark:text-slate-300">Integrate the final model into a production environment (a "data product") and monitor its performance over time.</dd>
            </div>
        </dl>
    </div>
);


const NumPyCheatSheet = () => (
  <div>
    <h3 className="text-xl font-semibold mb-3 text-sky-600 dark:text-sky-400">Array Creation</h3>
    <CodeBlock code={`
import numpy as np
arr = np.array([1, 2, 3])         # From a list
arr_range = np.arange(0, 10, 2)   # Like Python's range
arr_zeros = np.zeros((2, 3))      # Array of zeros
arr_ones = np.ones((3, 2))        # Array of ones
arr_rand = np.random.rand(2, 2)   # Random values (0 to 1)
    `} language="python" />

    <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Array Attributes & Inspection</h3>
    <CodeBlock code={`
arr = np.array([[1, 2], [3, 4]])
arr.shape  # (2, 2) -> Dimensions
arr.ndim   # 2 -> Number of dimensions
arr.dtype  # dtype('int64') -> Data type
arr.size   # 4 -> Total number of elements
    `} language="python" />
    
    <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Indexing & Slicing</h3>
    <CodeBlock code={`
arr = np.array([0, 10, 20, 30, 40])
arr[1]      # 10
arr[1:4]    # array([10, 20, 30])

arr2d = np.array([[1,2,3], [4,5,6]])
arr2d[0, 1] # 2 (Row 0, Col 1)
arr2d[:, 1] # array([2, 5]) (All rows, Col 1)
    `} language="python" />

    <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Math & Aggregations (Vectorization)</h3>
    <p>Operations are element-wise, avoiding slow Python loops. This is NumPy's key feature.</p>
    <CodeBlock code={`
arr = np.array([1, 2, 3])
arr + 5      # array([6, 7, 8]) -> Broadcasting
arr * 2      # array([2, 4, 6])
np.sum(arr)  # 6
np.mean(arr) # 2.0
np.std(arr)  # 0.816...
    `} language="python" />
  </div>
);

const PandasCheatSheet = () => (
    <div>
        <h3 className="text-xl font-semibold mb-3 text-sky-600 dark:text-sky-400">Data I/O (PYP 2024)</h3>
        <CodeBlock code={`
import pandas as pd
# Reading a CSV is a common first step
df = pd.read_csv('my_file.csv') 

df.to_csv('output.csv', index=False) # index=False prevents writing row numbers
# To read Excel, install openpyxl: pip install openpyxl
df_excel = pd.read_excel('my_file.xlsx')
        `} language="python" />
        
        <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Inspection & Exploration (PYP 2024)</h3>
        <ul className="list-disc list-inside space-y-2">
            <li><code className="font-mono bg-slate-100 dark:bg-slate-700 p-1 rounded-md">df.head(n)</code>: View first 'n' rows.</li>
            <li><code className="font-mono bg-slate-100 dark:bg-slate-700 p-1 rounded-md">df.describe()</code>: Get summary statistics for numerical columns.</li>
            <li><code className="font-mono bg-slate-100 dark:bg-slate-700 p-1 rounded-md">df.info()</code>: Column data types, memory usage, and non-null counts.</li>
            <li><code className="font-mono bg-slate-100 dark:bg-slate-700 p-1 rounded-md">df.shape</code>: Get (number of rows, number of columns).</li>
            <li><code className="font-mono bg-slate-100 dark:bg-slate-700 p-1 rounded-md">df['col'].value_counts()</code>: Count unique values in a Series.</li>
        </ul>

        <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Selection & Indexing</h3>
        <CodeBlock code={`
# Select columns
df['col_name']      # Select one column (returns a Series)
df[['col1', 'col2']] # Select multiple columns (returns a DataFrame)

# Boolean indexing
df[df['age'] > 30]
df[(df['age'] > 30) & (df['city'] == 'New York')]

# Selection by label (.loc) - INCLUSIVE
df.loc[0:5, ['col1', 'col2']] 

# Selection by position (.iloc) - EXCLUSIVE
df.iloc[0:5, 0:2]
        `} language="python" />

        <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Data Cleaning (PYP 2024)</h3>
        <CodeBlock code={`
df.isnull().sum()             # Count missing values per column
df.dropna()                   # Drop rows with any missing values
df.fillna(value)              # Fill missing values with 'value'
df.drop_duplicates()          # Remove duplicate rows
df.drop(columns=['col_name']) # Remove a specific column
        `} language="python" />
        
        <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Grouping & Aggregation</h3>
        <CodeBlock code={`
# Group by a column and calculate mean salary for each group
df.groupby('department')['salary'].mean()

# Group by multiple columns and calculate multiple aggregations
df.groupby(['department', 'region']).agg({
    'salary': 'mean',
    'sales': 'sum'
})
        `} language="python" />

        <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Sorting & Combining (PYP 2024)</h3>
         <CodeBlock code={`
# Sorting by values
df.sort_values(by='Income', ascending=False)

# Appends rows of df2 to df1
pd.concat([df1, df2])

# SQL-style join on a key column
pd.merge(df1, df2, on='key_column', how='inner')
# 'how' can be 'inner', 'left', 'right', 'outer'
        `} language="python" />
    </div>
);

const DataVizCheatSheet = () => (
    <div>
        <h3 className="text-xl font-semibold mb-3 text-sky-600 dark:text-sky-400">Matplotlib (Object-Oriented API)</h3>
        <p>Provides full control over your plots. Best practice for complex visualizations.</p>
        <CodeBlock code={`
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

# 1. Create Figure and Axes objects
fig, ax = plt.subplots()

# 2. Plot data on the Axes
ax.plot(x, y, label='sin(x)')

# 3. Customize the plot
ax.set_title('Sine Wave')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.grid(True)
ax.legend()

# 4. Show the plot
plt.show()
        `} language="python" />

        <h3 className="text-xl font-semibold mt-6 mb-3 text-sky-600 dark:text-sky-400">Seaborn (High-Level Statistical Plots)</h3>
        <p>Built on top of Matplotlib, Seaborn provides a high-level interface for drawing attractive statistical graphics and integrates well with Pandas DataFrames.</p>
        <CodeBlock code={`
import seaborn as sns
import matplotlib.pyplot as plt

# Histogram / Distribution Plot
sns.histplot(data=df, x='age', kde=True)
plt.title('Age Distribution')
plt.show()

# Scatter Plot
sns.scatterplot(data=df, x='income', y='spending_score', hue='gender')
plt.title('Income vs. Spending')
plt.show()

# Box Plot
sns.boxplot(data=df, x='category', y='value')
plt.title('Value Distribution by Category')
plt.show()
        `} language="python" />
    </div>
);

const ScikitLearnCheatSheet = () => (
  <div>
    <h3 className="text-xl font-semibold mb-3 text-sky-600 dark:text-sky-400">Core Machine Learning Workflow</h3>
    <p>A complete, end-to-end example of training a simple classification model.</p>
    <CodeBlock code={`
# 1. Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Assume 'X' is a DataFrame of features and 'y' is a Series for the target
# X = df[['feature1', 'feature2']]
# y = df['target']

# For demonstration, create dummy data:
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=10, random_state=42)


# 2. Split data into training and testing sets
# test_size=0.3 means 30% of data is for testing
# random_state ensures the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y # stratify is good practice
)


# 3. Initialize the model
# The API is consistent across models, so you can easily swap this line
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(random_state=42)
model = LogisticRegression()


# 4. Train the model on the training data
model.fit(X_train, y_train)


# 5. Make predictions on the unseen test data
predictions = model.predict(X_test)


# 6. Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# Confusion matrix shows correct vs. incorrect predictions
cm = confusion_matrix(y_test, predictions)
print("\\nConfusion Matrix:")
print(cm)
    `} language="python" />
  </div>
);


export const cheatSheetData: CheatSheetSection[] = [
    {
        title: 'Python Fundamentals & Data Structures (Exam Focus)',
        content: <PythonFundamentalsCheatSheet />
    },
    {
        title: 'The Data Science Pipeline',
        content: <DataSciencePipelineCheatSheet />
    },
    {
        title: 'Pandas Data Wrangling Cookbook',
        content: <PandasCheatSheet />
    },
    {
        title: 'NumPy Essentials',
        content: <NumPyCheatSheet />
    },
    {
        title: 'Data Visualization with Matplotlib & Seaborn',
        content: <DataVizCheatSheet />
    },
    {
        title: 'Scikit-Learn Machine Learning Workflow',
        content: <ScikitLearnCheatSheet />
    },
];
