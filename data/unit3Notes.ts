import React from 'react';
import { Unit } from '../types';
import { SQLJoinsDiagram } from '../components/Diagrams';

export const unit3Notes: Unit = {
    id: 3,
    title: "Unit 3: Getting Your Hands Dirty With Data",
    topics: [
        {
            title: "3.1 Working with Tools: Jupyter Notebook",
            content: `Jupyter Notebook is the quintessential tool for interactive data science in Python. It's a web-based environment that allows you to create and share documents containing live code, equations, visualizations, and narrative text. Its power lies in its interactive, cell-based workflow, which is ideal for data exploration, rapid prototyping, and documenting your analysis journey from start to finish.

- **3.1.1-3.1.4 Getting Started and Getting Help:**
    - **Architecture (Kernel and Client):** Jupyter operates on a two-part model. The **Kernel** is a separate process that runs your code (e.g., a Python kernel). The **Client** is the user interface in your web browser where you write and view your code, text, and output. They communicate with each other. This is why you must leave the terminal window running; closing it shuts down the kernel, and you lose all your variables.
    - **Interacting with Text (Markdown):** Cells can be set to 'Code' or 'Markdown'. Markdown is a simple syntax for formatting rich text, allowing you to create a narrative around your code. This is crucial for creating understandable and shareable reports.
    - **Getting Help:** Jupyter provides instant access to documentation. This is a superpower for efficient coding.
        - \`function_name?\`: Displays the function's docstring, which explains its purpose, parameters, and return value.
        - \`function_name??\`: Shows the full source code of the function, useful for understanding its implementation details.
        - **Tab Completion:** A huge time-saver. Pressing \`Tab\` after typing part of a variable or function name will autocomplete it or show a list of possible completions.

- **3.1.5 Magic Functions:** Special commands, not part of the Python language itself, that provide extra functionality. They are prefixed with \`%\` (line magic, applies to one line) or \`%%\` (cell magic, applies to the whole cell).
    - **\`%matplotlib inline\`:** The most common magic command. It ensures that Matplotlib plots are rendered directly within the notebook's output cell.
    - **\`%timeit <statement>\`:** Precisely measures the execution time of a single line of code by running it many times and providing a statistical average.
    - **\`%%time\`:** Measures the wall-clock time for an entire cell's code to execute. It only runs the code once.
    - **\`%who\`:** Lists all variables currently defined in the kernel's memory.

- **3.1.7-3.1.9 Key Features & Best Practices:**
    - **Kernel Management:** The kernel maintains the state of your notebook (variables, function definitions, etc.).
        - **Restart:** Clears all variables from memory and starts a fresh session. **Best Practice:** Periodically restart your kernel and run all cells from top to bottom (\`Kernel -> Restart & Run All\`). This is the gold standard for ensuring your code is reproducible and doesn't depend on variables you defined in a now-deleted cell.
        - **Interrupt:** Stops a cell that is taking too long to execute (e.g., an infinite loop).
    - **Checkpoints:** Jupyter automatically saves checkpoints, allowing you to revert to a previous state of your notebook if you make a mistake.`,
            code: `import numpy as np
import time

# Example of using '?' to get help in a real notebook
# You would uncomment the line below in a Jupyter cell to see the help text.
# np.linspace? 

# Example of comparing magic functions
# %timeit is great for fast operations
print("Using %timeit for a fast operation:")
%timeit np.random.rand(100, 100)

print("\\n" + "="*40 + "\\n")

# %%time is better for longer operations
print("Using %%time for a slower cell operation:")
%%time
time.sleep(1) # Simulate a long-running task
long_list = [x**2 for x in range(100000)]
del long_list`,
            output: `Using %timeit for a fast operation:
# Output will vary slightly depending on your machine
80.3 µs ± 1.3 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

========================================

Using %%time for a slower cell operation:
# Output shows the total time for the cell to run
CPU times: user 22.5 ms, sys: 3.4 ms, total: 25.9 ms
Wall time: 1.03 s`
        },
        {
            title: "3.2 Working with Real Data: Uploading, Streaming, and Sampling",
            content: `- **3.2.2 Uploading Small Data to Memory:** This is the most common approach for datasets that are smaller than your computer's available RAM (typically < 1-2 GB). You use a library like pandas to read the entire file into a DataFrame in one go. This is fast and convenient, as the entire dataset is instantly accessible for any operation.

- **3.2.3 Streaming Large Data:** For datasets that are too large to fit in memory, you must process them in smaller pieces or "chunks." The \`chunksize\` parameter in \`pd.read_csv()\` creates an iterator object. Instead of one large DataFrame, you get an object that you can loop through, yielding one smaller DataFrame (a chunk) at a time. In each iteration, you perform your calculations on the current chunk and then aggregate the result. This is a memory-efficient way to process massive files.

- **3.2.4 Data Augmentation for Image Data:** In computer vision, a model's performance depends heavily on the size and diversity of the training data. Data augmentation artificially expands the training set by creating modified copies of existing images. Common techniques include rotating, flipping, zooming, adjusting brightness, and adding noise. This process teaches the model to be robust to these variations (e.g., to recognize a cat whether it's centered or in the corner), which helps it generalize better to new, unseen images and significantly reduces overfitting.

- **3.2.5 Sampling Data Methods:** When exploring a massive dataset, running analyses on the full data can be slow. It's often more efficient to work with a smaller, representative sample.
    - **Simple Random Sampling:** Every data point has an equal chance of being selected. Ideal for quick exploration. Use \`df.sample(n=1000)\` for a fixed number or \`df.sample(frac=0.1)\` for a fraction.
    - **Stratified Sampling:** The population is divided into subgroups (strata), and random samples are drawn from each subgroup, preserving the original data's proportions. This is critical when dealing with imbalanced data (e.g., a fraud detection dataset where only 1% of transactions are fraudulent) to ensure the sample accurately reflects all categories.`,
            code: `import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split # Used for stratified sampling example

# --- Streaming Example ---
# Simulating a large CSV file with 1000 rows
large_csv_data = "\\n".join([f"{i},{i*10}" for i in range(1000)])
csv_file = StringIO(large_csv_data)
csv_file.seek(0) # Reset cursor to the beginning of the file-like object

# Process the file in chunks of 200 rows at a time
total_sum = 0
chunk_iterator = pd.read_csv(csv_file, header=None, chunksize=200)

print("--- Processing Large File in Chunks ---")
for i, chunk in enumerate(chunk_iterator):
    # 'chunk' is a DataFrame with 200 rows
    print(f"Processing chunk {i+1}...")
    total_sum += chunk[1].sum()
print(f"Total sum from streaming: {total_sum}")


# --- Stratified Sampling Example ---
# Create an imbalanced dataset: 90% class 'A', 10% class 'B'
data = {'feature': range(100), 'class': ['A']*90 + ['B']*10}
df = pd.DataFrame(data)

# Simple random sampling might miss the rare class 'B' entirely in a small sample
random_sample = df.sample(n=5, random_state=42)
print("\\n--- Simple Random Sample ---")
print(random_sample['class'].value_counts())

# Stratified sampling preserves the 9:1 ratio even in a small sample
_, stratified_sample = train_test_split(df, test_size=0.10, stratify=df['class'], random_state=42)
print("\\n--- Stratified Sample (10% of data) ---")
print(stratified_sample['class'].value_counts())`,
            output: `--- Processing Large File in Chunks ---
Processing chunk 1...
Processing chunk 2...
Processing chunk 3...
Processing chunk 4...
Processing chunk 5...
Total sum from streaming: 4995000

--- Simple Random Sample ---
class
A    5
Name: count, dtype: int64

--- Stratified Sample (10% of data) ---
class
A    9
B    1
Name: count, dtype: int64`
        },
        {
            title: "3.2.6 Accessing Data in Structured Flat-Files",
            content: `Flat files are the most common way to store and share data. Pandas provides excellent, highly optimized tools for reading them.
- **Reading CSV Files (.csv):** Comma-Separated Values. \`pandas.read_csv()\` is a powerful and flexible function. Key parameters include:
    - \`filepath_or_buffer\`: Path to the file or a URL.
    - \`sep\` (or \`delimiter\`): The character separating columns (e.g., \`','\`, \`'\\t'\` for tab, \`';'\`).
    - \`header\`: Row number to use as column names. Use \`header=None\` if the file has no header.
    - \`names\`: A list of column names to use, especially if \`header=None\`.
    - \`index_col\`: The column to use as the row labels (the index of the DataFrame).
    - \`usecols\`: A list of column names or indices to read. This is a great memory optimization if you only need a subset of columns.
    - \`na_values\`: A list of strings to recognize as missing values (e.g., ['NA', '--', 'Missing']).
    - \`parse_dates\`: A list of columns that should be parsed as dates.
- **Reading Excel Files (.xlsx):** Use \`pandas.read_excel()\`. You must first install a library like \`openpyxl\` (\`pip install openpyxl\`). The most important parameter is \`sheet_name\`, which lets you specify which sheet to read by name or index.
- **Reading JSON Files (.json):** JavaScript Object Notation is common for web APIs. \`pd.read_json()\` can read JSON files. The \`orient\` parameter is critical to match the JSON structure (e.g., 'records' for a list of dictionaries, 'columns' for a dictionary of lists).`,
            code: `import pandas as pd
from io import StringIO

# Example reading a more complex CSV with a different delimiter and no header
# We also want to parse dates and handle custom missing values
csv_data = "1|Alice|85|2023-01-05\\n2|Bob|--|2023-01-06\\n3|Charlie|78|2023-01-07"

df = pd.read_csv(
    StringIO(csv_data), 
    sep='|', 
    header=None,
    names=['ID', 'Name', 'Score', 'JoinDate'],
    na_values=['--'],
    parse_dates=['JoinDate'] # Tell pandas to convert this column to datetime objects
)

print(df)
print("\\n--- Info after reading ---")
# Note that 'Score' is now float64 (to accommodate NaN) and 'JoinDate' is datetime64
df.info()`,
            output: `   ID     Name  Score   JoinDate
0   1    Alice   85.0 2023-01-05
1   2      Bob    NaN 2023-01-06
2   3  Charlie   78.0 2023-01-07

--- Info after reading ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype         
---  ------    --------------  -----         
 0   ID        3 non-null      int64         
 1   Name      3 non-null      object        
 2   Score     2 non-null      float64       
 3   JoinDate  3 non-null      datetime64[ns]
dtypes: datetime64[ns](1), float64(1), int64(1), object(1)
memory usage: 228.0+ bytes`
        },
        {
            title: "3.2.7-3.2.9 Accessing Other Data Sources",
            content: `- **Relational Databases (SQL):** You can query SQL databases directly into a pandas DataFrame using \`pd.read_sql_query()\`. To connect to a database, you typically use the \`SQLAlchemy\` library, which provides a standard interface for connecting to various database types (PostgreSQL, MySQL, SQLite, etc.).

- **Web APIs:** Many websites provide Application Programming Interfaces (APIs) to access their data in a structured way (usually JSON). The \`requests\` library is the standard for making HTTP requests. The typical workflow is:
    1.  Use \`requests.get(url)\` to fetch the data.
    2.  Check \`response.status_code\` to ensure the request was successful (200 means OK).
    3.  Use \`response.json()\` to parse the JSON content into a Python dictionary or list.
    4.  Convert this into a DataFrame, often with \`pd.DataFrame()\`.

- **Web Scraping (HTML):** For websites without an API, you can extract data directly from their HTML.
    - **Pandas \`read_html()\`:** This powerful function acts as a blunt instrument. It scans a URL, finds all HTML \`<table>\` tags, and automatically converts each one into a DataFrame. It returns a list of all DataFrames found. This is incredibly efficient for grabbing tabular data.
    - **Beautiful Soup:** This library is a precision tool (a scalpel). It parses HTML into a navigable object, allowing you to find specific tags, classes, or IDs to extract the exact data you need, such as text from paragraphs, links, or specific items from a list.`,
            code: `import pandas as pd
import requests # You might need to run: pip install requests
import sqlite3

# --- SQL Example with SQLite (built-in) ---
# 1. Create a temporary in-memory database and table
conn = sqlite3.connect(':memory:')
c = conn.cursor()
c.execute('''CREATE TABLE stocks (date text, symbol text, price real)''')
c.execute("INSERT INTO stocks VALUES ('2023-01-01', 'AAPL', 150.0)")
c.execute("INSERT INTO stocks VALUES ('2023-01-01', 'GOOG', 2800.0)")
conn.commit()

# 2. Write a query and read data into a DataFrame
query = "SELECT * FROM stocks WHERE price > 200.0"
sql_df = pd.read_sql_query(query, conn)
conn.close()
print("--- Data from SQL Query ---")
print(sql_df)

# --- Web API Example ---
# A simple, free API for testing
api_url = "https://jsonplaceholder.typicode.com/users"
response = requests.get(api_url)
if response.status_code == 200:
    api_data = response.json() # Parse JSON into a list of dictionaries
    api_df = pd.DataFrame(api_data)
    print("\\n--- Data from Web API ---")
    # Show specific columns for brevity
    print(api_df[['id', 'name', 'email']].head())
else:
    print(f"\\nFailed to fetch API data. Status code: {response.status_code}")`,
            output: `--- Data from SQL Query ---
         date symbol   price
0  2023-01-01   GOOG  2800.0

--- Data from Web API ---
   id               name                          email
0   1       Leanne Graham          Sincere@april.biz
1   2        Ervin Howell           Shanna@melissa.tv
2   3  Clementine Bauch  Nathan@yesenia.net
3   4      Patricia Lebsack      Julianne.OConner@kory.org
4   5    Chelsey Dietrich   Lucio_Hettinger@annie.ca`
        },
        {
            title: "3.3.2-3.3.5 Conditioning: Validating Your Data",
            content: `Data validation is the process of auditing your data to ensure its quality and integrity before analysis. It's about asking, "Is my data what I think it is?" and is the first step in data cleaning.

- **3.3.3 Discovering Data Content (The Initial Audit):**
    - **\`df.info()\`:** Your first check. Provides a high-level summary. Look for:
        1.  **Dtype:** Are columns the correct data type? (e.g., dates read as \`object\`, numbers as \`object\`).
        2.  **Non-Null Count:** Get a quick sense of which columns have missing values.
    - **\`df.describe()\`:** For numerical columns. Generates descriptive statistics. Look for:
        1.  **min/max:** Are there illogical values? (e.g., an age of -5 or 500).
        2.  **std (Standard Deviation):** Is it 0? If so, the column is a constant value and provides no information.
        3.  **mean vs 50% (median):** Are they very different? This indicates the data is skewed.
    - **\`df['column'].value_counts()\`:** For categorical columns. Shows the frequency of each category. Look for:
        1.  **Inconsistent entries:** ('USA', 'U.S.A.', 'usa').
        2.  **Long tail:** Many categories with very few entries that might need to be grouped.
- **3.3.4 Removing Duplicates:** Duplicate records can bias your analysis. Use \`df.duplicated().sum()\` to check for fully duplicate rows and \`df.drop_duplicates()\` to remove them. Often, duplicates are defined by a subset of columns (e.g., same user ID and timestamp). Use \`df.drop_duplicates(subset=['user_id', 'timestamp'])\`. The \`keep\` parameter (\`'first'\`, \`'last'\`, \`False\`) determines which duplicate to keep.
- **3.3.5 Creating a Data Map/Plan:** Before changing anything, document your plan. For each column, note its expected data type, how you will handle missing values, and any transformations needed. This ensures a systematic and reproducible cleaning process.`,
            code: `import pandas as pd
import numpy as np
from io import StringIO

# Create a messy dataset to diagnose
messy_data = """user_id,age,city,signup_date
1,25,New York,2023-01-10
2,31,london,2023-01-11
3,-15,New York,2023-01-12
4,40,chicago,2023-01-13
2,31,london,2023-01-11
5,,chicago,2023-01-15
"""
df = pd.read_csv(StringIO(messy_data))

# --- Diagnosis ---
print("--- df.info() ---")
df.info() # Age has a missing value, signup_date is an object

print("\\n--- df.describe() ---")
print(df.describe()) # Age has a min of -15, which is impossible

print("\\n--- City Value Counts ---")
print(df['city'].value_counts()) # Inconsistent capitalization

print(f"\\nNumber of duplicate rows: {df.duplicated().sum()}")`,
            output: `--- df.info() ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 4 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   user_id      6 non-null      int64  
 1   age          5 non-null      float64
 2   city         6 non-null      object 
 3   signup_date  6 non-null      object 
dtypes: float64(1), int64(1), object(2)
memory usage: 320.0+ bytes

--- df.describe() ---
       user_id        age
count  6.000000   5.000000
mean   2.833333  22.400000
std    1.471960  22.153916
min    1.000000 -15.000000
25%    2.000000  25.000000
50%    2.500000  31.000000
75%    3.750000  31.000000
max    5.000000  40.000000

--- City Value Counts ---
city
New York    2
london      2
chicago     2
Name: count, dtype: int64

Number of duplicate rows: 1`
        },
        {
            title: "3.3.6 Conditioning: Manipulating Categorical Variables",
            content: `Most machine learning models require numerical input, so categorical data (text labels) must be converted into a numerical format. This is called encoding. The strategy depends on whether the data is nominal or ordinal.

- **Nominal Data:** Categories with no intrinsic order (e.g., 'Color', 'City').
    - **One-Hot Encoding:** The safest and most common strategy. It creates a new binary (0 or 1) column for each category. For a 'Color' column with 'Red', 'Green', 'Blue', it creates three new columns: \`is_Red\`, \`is_Green\`, \`is_Blue\`. This avoids implying any false order between colors. Pandas provides \`pd.get_dummies()\` for this.
    - **Dummy Variable Trap:** When using one-hot encoding, the new columns are perfectly multicollinear. To avoid this, which can be an issue for some models, you can drop one of the new columns by using \`pd.get_dummies(..., drop_first=True)\`.

- **Ordinal Data:** Categories with a clear, meaningful order (e.g., 'Small' < 'Medium' < 'Large').
    - **Label Encoding:** Assigns a unique integer to each category (e.g., 'Small' -> 0, 'Medium' -> 1, 'Large' -> 2). This preserves the ordinal relationship and is computationally efficient.
    - **Custom Mapping:** The most explicit way is to define a dictionary that maps each category to a number and use the \`.map()\` method. This gives you full control over the encoding.`,
            code: `import pandas as pd

df = pd.DataFrame({
    'color': ['Red', 'Green', 'Blue', 'Green'], # Nominal data
    'size': ['S', 'M', 'L', 'S'] # Ordinal data
})

# --- One-Hot Encode the 'color' column (nominal data) ---
# drop_first=True is good practice to avoid multicollinearity
one_hot_df = pd.get_dummies(df, columns=['color'], prefix='c', drop_first=True)
print("--- One-Hot Encoded ---")
print(one_hot_df)

# --- Ordinal Encode the 'size' column using a custom map ---
size_mapping = {'S': 0, 'M': 1, 'L': 2}
df['size_encoded'] = df['size'].map(size_mapping)
print("\\n--- Ordinal Encoded (with mapping) ---")
print(df[['size', 'size_encoded']])`,
            output: `--- One-Hot Encoded ---
  size  c_Green  c_Red
0    S        0      1
1    M        1      0
2    L        0      0
3    S        1      0

--- Ordinal Encoded (with mapping) ---
  size  size_encoded
0    S             0
1    M             1
2    L             2
3    S             0`
        },
        {
            title: "3.3.8 Conditioning: Dealing with Missing Data",
            content: `Missing values (\`NaN\`, \`None\`, etc.) are unavoidable. How you handle them can significantly impact your model's performance.

- **Finding Missing Data:** Use \`df.isnull().sum()\` to get a count of missing values per column.

- **Strategies for Handling Missing Data:**
    - **Dropping:** \`df.dropna()\` removes rows (or columns) with missing data. This is simple but can cause significant data loss. It's generally only safe if the percentage of missing data is very small (<5%) and the data is missing completely at random.
    - **Imputing with a Constant:** Filling with a fixed value like 0, -1, or 'Unknown'. This can be effective if the missingness itself is meaningful (e.g., a missing 'Date_Returned' value means the item hasn't been returned).
    - **Imputing with Statistics (Mean/Median/Mode):** This is a very common baseline strategy.
        - **Mean:** Use for numerical columns that have a symmetric distribution (like a bell curve).
        - **Median:** Use for numerical columns that are skewed by outliers (e.g., 'Income'). The median is more robust.
        - **Mode:** Use for categorical columns. Fill with the most frequent category.
    - **Advanced Imputation:** More complex methods exist, such as using a machine learning model (like K-Nearest Neighbors) to predict the most likely value for a missing entry based on its other features.`,
            code: `import pandas as pd
import numpy as np
df = pd.DataFrame({
    'Age': [25, 30, np.nan, 45, 30], # Skewed if we had more data, let's use median
    'City': ['NY', 'SF', 'NY', np.nan, 'NY'] # Categorical, use mode
})
print("--- Original Data ---")
print(df)
print("\\nMissing values before imputation:")
print(df.isnull().sum())


# Impute 'Age' with the median (robust to outliers)
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)

# Impute 'City' with the mode (most frequent value)
city_mode = df['City'].mode()[0]
df['City'].fillna(city_mode, inplace=True)

print("\\n--- Imputed Data ---")
print(df)`,
            output: `--- Original Data ---
    Age City
0  25.0   NY
1  30.0   SF
2   NaN   NY
3  45.0  NaN
4  30.0   NY

Missing values before imputation:
Age     1
City    1
dtype: int64

--- Imputed Data ---
    Age City
0  25.0   NY
1  30.0   SF
2  30.0   NY
3  45.0   NY
4  30.0   NY`
        },
        {
            title: "3.3.9 Conditioning: Slicing and Dicing (Indexing)",
            content: `This refers to selecting specific subsets of your data for analysis. Pandas offers several powerful methods.

- **Selecting Columns:**
    - Single column: \`df['Col1']\` (returns a Series).
    - Multiple columns: \`df[['Col1', 'Col2']]\` (returns a DataFrame).

- **Filtering Rows (Boolean Indexing):** The most common way to select rows that meet a condition. Create a boolean mask and pass it to the DataFrame: \`df[df['Age'] > 30]\`. Combine conditions with \`&\` (and) and \`|\` (or), wrapping each condition in parentheses due to operator precedence.

- **\`.loc\` and \`.iloc\` (The Preferred Indexers):** These are more explicit and robust than basic indexing, especially for assignment.
    - **\`.loc\` (Label-based):** Selects data by its index and column labels. It is inclusive of the endpoint in slices. Example: \`df.loc[start_label:end_label, 'Column_Name']\`.
    - **\`.iloc\` (Integer-based):** Selects data by its integer position. It is exclusive of the endpoint in slices, like standard Python lists. Example: \`df.iloc[start_pos:end_pos, column_pos]\`.

- **Warning on Chained Indexing:** Avoid assigning values using chained indexing like \`df['age'][0] = 26\`. This can fail unpredictably. The correct, guaranteed way is to use a single accessor: \`df.loc[0, 'age'] = 26\`.`,
            code: `import pandas as pd
df = pd.DataFrame({
    'age': [25, 30, 35, 40], 
    'city': ['NY', 'SF', 'LA', 'SF'],
    'score': [88, 92, 95, 90]
}, index=['a', 'b', 'c', 'd'])

# --- Boolean Indexing ---
# Find people from SF with a score > 90
mask = (df['city'] == 'SF') & (df['score'] > 90)
print("--- Boolean Indexing ---")
print(df[mask])

# --- .loc example ---
# Select rows with index 'a' through 'c' and the 'age' and 'score' columns
print("\\n--- .loc Example ---")
print(df.loc['a':'c', ['age', 'score']])

# --- .iloc example ---
# Select the first two rows (position 0 and 1) and the first two columns (position 0 and 1)
print("\\n--- .iloc Example ---")
print(df.iloc[0:2, 0:2])`,
            output: `--- Boolean Indexing ---
   age city  score
b   30   SF     92

--- .loc Example ---
   age  score
a   25     88
b   30     92
c   35     95

--- .iloc Example ---
   age city
a   25   NY
b   30   SF`
        },
        {
            title: "3.3.10-3.3.12 Conditioning: Concatenating, Merging, and Transforming",
            content: `- **Concatenating (\`pd.concat\`):** This function "stacks" DataFrames together. It is useful for combining datasets that have the same columns.
    - \`axis=0\` (default): Stacks vertically (appends rows). Use \`ignore_index=True\` to reset the index of the combined DataFrame.
    - \`axis=1\`: Stacks horizontally (appends columns), aligning on the index.
- **Merging (\`pd.merge\`):** This is the primary tool for combining DataFrames based on common columns, similar to a SQL JOIN. Key parameters:
    - \`left\` and \`right\`: The two DataFrames to merge.
    - \`on\`: The column name(s) to join on.
    - \`how\`: The type of merge:
        - \`'inner'\` (default): Keeps only rows with matching keys in both DataFrames.
        - \`'left'\`: Keeps all rows from the left DataFrame and matching rows from the right.
        - \`'right'\`: Keeps all rows from the right DataFrame and matching rows from the left.
        - \`'outer'\`: Keeps all rows from both DataFrames.
- **Sorting and Shuffling:**
    - \`df.sort_values(by='column_name')\` is used to order your data. Use \`ascending=False\` for descending order.
    - \`df.sample(frac=1).reset_index(drop=True)\` randomly shuffles all rows. This is a critical step before splitting data into training and testing sets to ensure there is no inherent order.`,
            diagram: {
                component: React.createElement(SQLJoinsDiagram),
                caption: 'Visualizing different SQL join types (Inner, Left, Right, Outer) using Venn diagrams to show which records are kept.',
            },
            code: `import pandas as pd
employees = pd.DataFrame({'emp_id': ['E1', 'E2', 'E3'], 'name': ['Alice', 'Bob', 'Charlie']})
salaries = pd.DataFrame({'emp_id': ['E1', 'E2', 'E4'], 'salary': [70000, 80000, 90000]})

# --- Inner Join ---
# Only keeps employees E1 and E2, who are in both tables
inner_join = pd.merge(employees, salaries, on='emp_id', how='inner')
print("--- Inner Join ---")
print(inner_join)

# --- Left Join ---
# Keeps all employees from the 'employees' table
left_join = pd.merge(employees, salaries, on='emp_id', how='left')
print("\\n--- Left Join ---")
print(left_join)

# --- Outer Join ---
# Keeps all employees from both tables
outer_join = pd.merge(employees, salaries, on='emp_id', how='outer')
print("\\n--- Outer Join ---")
print(outer_join)`,
            output: `--- Inner Join ---
  emp_id   name  salary
0     E1  Alice   70000
1     E2    Bob   80000

--- Left Join ---
  emp_id     name   salary
0     E1    Alice  70000.0
1     E2      Bob  80000.0
2     E3  Charlie      NaN

--- Outer Join ---
  emp_id     name   salary
0     E1    Alice  70000.0
1     E2      Bob  80000.0
2     E3  Charlie      NaN
3     E4      NaN  90000.0`
        },
        {
            title: "3.3.13 Conditioning: Aggregating Data",
            content: `Aggregation involves collapsing many data points into a summary statistic. This is the core of exploratory data analysis.
- **The \`.groupby()\` Method:** This enables the powerful "Split-Apply-Combine" strategy:
    1.  **Split:** The data is split into groups based on the unique values in one or more columns (e.g., group by 'Department').
    2.  **Apply:** An aggregation function (\`sum()\`, \`mean()\`, \`count()\`, \`nunique()\`, etc.) is applied to the data within each group.
    3.  **Combine:** The results are combined into a new data structure (usually a DataFrame or Series).
- **Multiple Aggregations (\`.agg()\`):** You can perform multiple aggregations at once by passing a list of functions or a dictionary to the \`.agg()\` method. This allows you to calculate different statistics for different columns in a single, efficient step.
- **Pivot Tables (\`pd.pivot_table\`):** This is a flexible way to create a spreadsheet-style pivot table. It's an alternative way to perform grouping and aggregation, often creating a grid-like view that is easy to read. It requires an \`index\`, \`columns\`, and \`values\` to aggregate.`,
            code: `import pandas as pd
df = pd.DataFrame({
    'Category': ['Fruit', 'Veg', 'Fruit', 'Veg', 'Fruit'], 
    'Region': ['North', 'South', 'South', 'North', 'North'], 
    'Sales': [100, 150, 120, 130, 90],
    'Quantity': [10, 12, 15, 8, 9]
})

# Group by 'Category' and 'Region' and calculate multiple aggregations
summary = df.groupby(['Category', 'Region']).agg(
    total_sales=('Sales', 'sum'),
    avg_quantity=('Quantity', 'mean'),
    num_transactions=('Sales', 'count')
).reset_index() # .reset_index() converts the grouped output back to a flat DataFrame

print("--- GroupBy Summary ---")
print(summary)

# Pivot Table equivalent for total sales
pivot = pd.pivot_table(df, values='Sales', index='Category', columns='Region', aggfunc='sum')
print("\\n--- Pivot Table Summary (Total Sales) ---")
print(pivot)`,
            output: `--- GroupBy Summary ---
  Category Region  total_sales  avg_quantity  num_transactions
0    Fruit  North          190           9.5                 2
1    Fruit  South          120          15.0                 1
2      Veg  North          130           8.0                 1
3      Veg  South          150          12.0                 1

--- Pivot Table Summary (Total Sales) ---
Region    North  South
Category              
Fruit     190.0  120.0
Veg       130.0  150.0`
        },
        {
            title: "3.4.4 Shaping: Working with Raw Text",
            content: `Natural Language Processing (NLP) involves a preprocessing pipeline to convert unstructured text into a structured, numerical format suitable for analysis and modeling.

- **Text Normalization:** Cleaning the text. This usually involves converting all text to lowercase and removing punctuation, numbers, and any other non-essential characters like HTML tags.
- **Tokenization:** Splitting the cleaned text into a list of individual words or "tokens". This is the basic unit for most NLP analysis.
- **Stop Word Removal:** Removing common words (e.g., 'a', 'the', 'is', 'in') that add little semantic meaning. Libraries like NLTK and SpaCy provide predefined lists of stop words for many languages.
- **Stemming vs. Lemmatization:** Techniques for reducing words to their root form.
    - **Stemming:** A crude, rule-based process that chops off the end of words (e.g., 'studies', 'studying' -> 'studi'). It's fast but can result in non-dictionary words.
    - **Lemmatization:** A more sophisticated, dictionary-based process that returns the base form of a word, known as the "lemma" (e.g., 'studies', 'studying' -> 'study'). It is more accurate but slower. Lemmatization is usually preferred for better results.
- **Vectorization (TF-IDF):** After preprocessing, you must convert the text tokens into numerical vectors. Term Frequency-Inverse Document Frequency (TF-IDF) is a classic technique that measures how important a word is to a document in a collection of documents (a corpus). It gives higher weight to words that are frequent in one document but rare across all other documents, helping to highlight key terms.`,
            code: `# You may need to run: pip install nltk scikit-learn
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# One-time downloads for NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """A complete text preprocessing pipeline."""
    # 1. Lowercase and remove punctuation
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # 2. Tokenize
    tokens = word_tokenize(text)
    
    # 3. Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # 4. Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return " ".join(lemmas) # Return a cleaned string for the vectorizer

# --- Example Usage ---
corpus = [
    "The data scientists are studying different models.",
    "Machine learning models require clean data.",
    "Data preprocessing is an important step."
]

processed_corpus = [preprocess_text(doc) for doc in corpus]
print("--- Processed Corpus ---")
print(processed_corpus)

# --- Vectorization with TF-IDF ---
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_corpus)

# Display the results
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
print("\\n--- TF-IDF Matrix ---")
print(tfidf_df.round(2))`,
            output: `--- Processed Corpus ---
['data scientist studying different model', 'machine learning model require clean data', 'data preprocessing important step']

--- TF-IDF Matrix ---
   clean  data  different  important  learning  machine  model  preprocessing  require  scientist  step  studying
0   0.00  0.31       0.53       0.00      0.00     0.00   0.31           0.00     0.00       0.53  0.00      0.53
1   0.53  0.31       0.00       0.00      0.53     0.53   0.31           0.00     0.53       0.00  0.00      0.00
2   0.00  0.35       0.00       0.59      0.00     0.00   0.00           0.59     0.00       0.00  0.59      0.00`
        }
    ]
};