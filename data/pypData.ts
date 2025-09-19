import React from 'react';
import { PYPaper } from '../types';
import { DataSciencePipelineDiagram } from '../components/Diagrams';

export const pypData: PYPaper[] = [
  {
    year: '2022',
    questions: [
      {
        id: '22-Q1A',
        question: 'A. Compare list and set in python based on their characteristic.',
        answer: {
          theory: `
Lists and Sets are both fundamental data structures in Python used for storing collections of items, but they have distinct characteristics that make them suitable for different tasks. The primary differences lie in their ordering, mutability, and handling of duplicate elements.

**Here is a detailed comparison:**

| Characteristic      | List                                                              | Set                                                                      |
|---------------------|-------------------------------------------------------------------|--------------------------------------------------------------------------|
| Syntax          | Defined using square brackets: <br> \`my_list = [1, 'a', 2, 'a']\`  | Defined using curly braces or the \`set()\` constructor: <br> \`my_set = {1, 'a', 2}\` |
| Ordering        | Ordered: The order of elements is preserved. Items are stored and retrieved in the sequence they were added. | Unordered: Elements have no specific order. The order can change and should not be relied upon. |
| Indexing/Slicing| Supported: Elements can be accessed via numerical indices (e.g., \`my_list[0]\`). Slicing is also supported. | Not Supported: Because sets are unordered, you cannot access elements by an index. |
| Duplicates      | Allowed: A list can contain multiple identical elements.          | Not Allowed: Duplicates are automatically removed. Each element in a set must be unique. |
| Mutability      | Mutable: You can add, remove, or change elements after the list has been created. | Mutable: You can add or remove elements from a set. However, the elements themselves must be of an immutable type. |
| Performance     | Membership testing (\`x in my_list\`) is slow (O(n)) as it may require scanning the entire list. | Membership testing (\`x in my_set\`) is very fast (O(1) on average) due to its hash-based implementation. |
| Typical Use Case| Use when the order of items is important, and you need to store duplicates or access items by their position. | Use when you need to store a collection of unique items and perform fast membership checks or mathematical set operations (union, intersection). |

**Summary:**
- Choose a List when you need an ordered sequence that may contain duplicates, and you need to access elements by their position.
- Choose a Set when you need to ensure all elements are unique, order is not important, and you need to perform very fast checks for the existence of an element.
          `,
        },
      },
      {
        id: '22-Q1B',
        question: 'B. Demonstrate the concept of type casting using input function.',
        answer: {
          theory: `
**Concept of Type Casting**

In Python, type casting is the process of converting a variable from one data type to another. This is an explicit conversion performed by the programmer.

**Interaction with the \`input()\` Function**

The \`input()\` function is used to get data from the user through the console. A crucial characteristic of \`input()\` is that it always returns the user's input as a string (\`str\`) data type, regardless of what the user actually types.

For example, if the user enters the number \`25\`, the \`input()\` function does not return the integer \`25\`; it returns the string \`'25'\`.

**Why is Type Casting Necessary?**

This behavior becomes a problem when you need to perform mathematical or numerical operations. You cannot perform arithmetic on strings. For instance, trying to add 10 to the string \`'25'\` would result in a \`TypeError\` because Python doesn't know how to add an integer to a string.

To solve this, we must type cast the string received from the \`input()\` function into a numerical type, such as an integer (\`int\`) or a floating-point number (\`float\`), before using it in calculations.

**Demonstration Steps:**
1.  Read data from the user using \`input()\`.
2.  The returned value is stored in a variable, which will be of type \`str\`.
3.  Use casting functions like \`int()\` or \`float()\` to convert the string to the desired numeric type.
4.  Since the user might enter non-numeric text, this conversion can fail and raise a \`ValueError\`. It is best practice to wrap the type casting in a \`try-except\` block to handle such errors gracefully.
5.  Perform the numerical operations with the new, correctly-typed variable.
          `,
          code: `# Step 1: Get user input (which will be stored as strings)
age_str = input("Please enter your age: ")
weight_str = input("Please enter your weight in kg: ")

print(f"\\nData type of age input: {type(age_str)}")
print(f"Data type of weight input: {type(weight_str)}")

# Step 2: Use a try-except block to handle potential conversion errors
try:
    # Step 3: Type cast the string inputs to numeric types
    age_int = int(age_str)       # Cast to integer
    weight_float = float(weight_str) # Cast to float

    # Step 4: Perform calculations with the newly casted variables
    age_in_10_years = age_int + 10
    
    # Display the results and their new types
    print(f"\\nYour current age is {age_int} (type: {type(age_int)})")
    print(f"Your weight is {weight_float} kg (type: {type(weight_float)})")
    print(f"In 10 years, you will be {age_in_10_years} years old.")

except ValueError:
    # This block executes if the user enters non-numeric input
    print("\\nError: Invalid input. Please enter only numeric values for age and weight.")`,
          output: `Assuming user enters '25' for age and '70.5' for weight:

Please enter your age: 25
Please enter your weight in kg: 70.5

Data type of age input: <class 'str'>
Data type of weight input: <class 'str'>

Your current age is 25 (type: <class 'int'>)
Your weight is 70.5 kg (type: <class 'float'>)
In 10 years, you will be 35 years old.`,
        },
      },
      {
        id: '22-Q1C',
        question: `C. Consider the given details of the smart phone – IMEI_no, Phone_name, Company, Price, and Manufacturing_year. Perform following operation on it
1) Create five different dictionary objects of the smart phone.
2) Make a list object from the above created dictionary objects.
3) Sort the given list of objects according to the descending order of price of the smart phone and display it.`,
        answer: {
          theory: `
**Explanation of Concepts and Approach**

This problem requires us to model real-world objects (smartphones) using Python data structures and then perform a sorting operation. The choice of data structures is key to solving this efficiently.

**1. Creating Dictionary Objects:**
   - A Python dictionary is the ideal choice to represent a single smartphone. A dictionary stores data in key-value pairs, which maps perfectly to the smartphone's attributes. For example, the key \`'Phone_name'\` can be associated with the value \`'iPhone 13'\`. This makes the data structured and easy to read.

**2. Making a List of Dictionaries:**
   - To store a collection of these smartphone objects, a list is the most suitable data structure. A list is an ordered collection of items. By creating a list of dictionaries, we can manage multiple records, where each item in the list is a complete dictionary representing one smartphone.

**3. Sorting the List of Dictionaries:**
   - Python's built-in \`sorted()\` function is used to sort iterables like lists. However, when sorting a list of complex objects like dictionaries, we need to tell the function *which* piece of information to sort by.
   - The \`key\` parameter: This parameter accepts a function that is applied to each element in the list before comparison. The return value of this function is then used as the sorting key.
   - Lambda Function: We use a simple, anonymous function called a lambda function for this purpose. The expression \`key=lambda phone: phone['Price']\` tells the \`sorted()\` function: "For each element (which we'll call \`phone\`) in the list, get the value associated with the key \`'Price'\` and use that value for sorting."
   - The \`reverse\` parameter: To sort in descending order (highest price first), we set \`reverse=True\`.
          `,
          code: `import pprint # Used for pretty-printing the output to make it more readable

# 1) Create five different dictionary objects of the smart phone
# Each dictionary represents one smartphone with its attributes as key-value pairs.
phone1 = {'IMEI_no': '12345A', 'Phone_name': 'iPhone 13', 'Company': 'Apple', 'Price': 799, 'Manufacturing_year': 2021}
phone2 = {'IMEI_no': '67890B', 'Phone_name': 'Galaxy S22', 'Company': 'Samsung', 'Price': 699, 'Manufacturing_year': 2022}
phone3 = {'IMEI_no': '13579C', 'Phone_name': 'Pixel 6', 'Company': 'Google', 'Price': 599, 'Manufacturing_year': 2021}
phone4 = {'IMEI_no': '24680D', 'Phone_name': 'Nord 2', 'Company': 'OnePlus', 'Price': 399, 'Manufacturing_year': 2021}
phone5 = {'IMEI_no': '97531E', 'Phone_name': 'iPhone 14 Pro', 'Company': 'Apple', 'Price': 999, 'Manufacturing_year': 2022}

# 2) Make a list object from the above created dictionary objects
# This list now holds all the smartphone records.
smartphones = [phone1, phone2, phone3, phone4, phone5]
print("--- Original List of Smartphones ---")
pprint.pprint(smartphones)

# 3) Sort the list in descending order of price
# - sorted() creates a new sorted list.
# - key=lambda phone: phone['Price'] specifies that sorting should be based on the 'Price' value of each dictionary.
# - reverse=True specifies descending order.
sorted_smartphones = sorted(smartphones, key=lambda phone: phone['Price'], reverse=True)

print("\\n--- Sorted List of Smartphones by Price (Descending) ---")
pprint.pprint(sorted_smartphones)`,
          output: `--- Original List of Smartphones ---
[{'Company': 'Apple',
  'IMEI_no': '12345A',
  'Manufacturing_year': 2021,
  'Phone_name': 'iPhone 13',
  'Price': 799},
 {'Company': 'Samsung',
  'IMEI_no': '67890B',
  'Manufacturing_year': 2022,
  'Phone_name': 'Galaxy S22',
  'Price': 699},
 {'Company': 'Google',
  'IMEI_no': '13579C',
  'Manufacturing_year': 2021,
  'Phone_name': 'Pixel 6',
  'Price': 599},
 {'Company': 'OnePlus',
  'IMEI_no': '24680D',
  'Manufacturing_year': 2021,
  'Phone_name': 'Nord 2',
  'Price': 399},
 {'Company': 'Apple',
  'IMEI_no': '97531E',
  'Manufacturing_year': 2022,
  'Phone_name': 'iPhone 14 Pro',
  'Price': 999}]

--- Sorted List of Smartphones by Price (Descending) ---
[{'Company': 'Apple',
  'IMEI_no': '97531E',
  'Manufacturing_year': 2022,
  'Phone_name': 'iPhone 14 Pro',
  'Price': 999},
 {'Company': 'Apple',
  'IMEI_no': '12345A',
  'Manufacturing_year': 2021,
  'Phone_name': 'iPhone 13',
  'Price': 799},
 {'Company': 'Samsung',
  'IMEI_no': '67890B',
  'Manufacturing_year': 2022,
  'Phone_name': 'Galaxy S22',
  'Price': 699},
 {'Company': 'Google',
  'IMEI_no': '13579C',
  'Manufacturing_year': 2021,
  'Phone_name': 'Pixel 6',
  'Price': 599},
 {'Company': 'OnePlus',
  'IMEI_no': '24680D',
  'Manufacturing_year': 2021,
  'Phone_name': 'Nord 2',
  'Price': 399}]`,
        },
      },
      {
        id: '22-Q2A',
        question: 'A. Illustrate the concept of data science pipeline.',
        answer: {
          diagram: {
            component: React.createElement(DataSciencePipelineDiagram),
            caption: 'A flowchart illustrating the six key stages of the Data Science Pipeline, from problem definition to deployment.',
          },
          theory: `
The Data Science Pipeline (also known as the Data Science Lifecycle or Workflow) is a systematic, structured framework that outlines the key stages of a data science project, from its inception to its deployment. Following this pipeline ensures that projects are well-organized, reproducible, and effectively address the business problem.

**Here are the core stages of the pipeline:**

**1. Problem Definition & Data Acquisition:**
   - **Problem Definition:** This is the most crucial step. It involves collaborating with stakeholders to understand the business objective and translate it into a specific, measurable data science question. For example, "Can we predict which customers are likely to stop using our service?"
   - **Data Acquisition:** Once the problem is defined, the required data is gathered. This can come from various sources such as:
     - Internal databases (SQL)
     - Flat files (CSV, Excel)
     - External sources via APIs
     - Web scraping

**2. Data Preparation and Cleaning (Data Wrangling):**
   - Raw data is almost always messy, incomplete, and inconsistent. This stage, often the most time-consuming, focuses on making the data usable. Key tasks include:
     - **Handling Missing Values:** Deciding whether to remove records with missing data or to fill them in (impute) using techniques like mean, median, or more advanced models.
     - **Correcting Data Types:** Ensuring numerical columns are treated as numbers and dates are treated as datetime objects.
     - **Removing Duplicates:** Eliminating redundant records that could bias the analysis.
     - **Data Transformation:** Standardizing units, fixing typos, and creating new variables (feature engineering).

**3. Exploratory Data Analysis (EDA):**
   - This is the investigative phase where the goal is to "get to know" the data. Data scientists use descriptive statistics and visualizations to uncover patterns, identify anomalies, and form hypotheses.
     - **Univariate Analysis:** Analyzing single variables using histograms (for numerical data) and bar charts (for categorical data) to understand their distributions.
     - **Bivariate Analysis:** Analyzing the relationship between two variables using scatter plots (numerical vs. numerical) or box plots (numerical vs. categorical).
     - **Correlation Analysis:** Identifying which variables are related to each other.

**4. Modeling and Machine Learning:**
   - In this stage, a machine learning model is selected, trained, and evaluated to solve the defined problem.
     - **Feature Engineering:** Creating new input variables from existing ones to improve model performance.
     - **Model Selection:** Choosing an appropriate algorithm (e.g., Linear Regression for prediction, Random Forest for classification).
     - **Train-Test Split:** The data is divided into a training set (for the model to learn from) and a testing set (to evaluate its performance on unseen data). This helps prevent overfitting.
     - **Model Training:** The algorithm learns patterns from the training data.
     - **Model Evaluation:** The model's performance is measured on the test set using metrics like accuracy (for classification) or Mean Squared Error (for regression).

**5. Visualization and Communication of Results:**
   - The findings must be communicated effectively to stakeholders, who may not have a technical background.
     - **Data Storytelling:** Creating a compelling narrative that explains the results, the insights discovered, and their business implications.
     - **Explanatory Visualizations:** Using clear and simple charts (e.g., bar charts, line graphs) to highlight the key takeaways.

**6. Deployment & Monitoring:**
   - The final model is integrated into a production environment where it can provide real-time value. This is known as creating a data product.
     - **Deployment:** For example, embedding a recommendation model into an e-commerce website.
     - **Monitoring:** Continuously monitoring the model's performance to ensure it remains accurate over time and retraining it as new data becomes available.
`
        }
      },
    ],
  },
  {
    year: '2024',
    questions: [
      {
        id: '24-Q1A',
        question: 'A. Compare list and dictionary in python based on their characteristic.',
        answer: {
          theory: `
Lists and Dictionaries are two of the most commonly used built-in data structures in Python. While both are used to store collections of data and are mutable, they differ fundamentally in their structure, how they are accessed, and their intended use cases.

**Here is a detailed comparison:**

| Characteristic     | List                                                                 | Dictionary                                                                      |
|--------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Structure      | An ordered collection of items. The sequence is maintained.      | An unordered collection of key-value pairs. (Note: Ordered since Python 3.7) |
| Access         | Elements are accessed by their numerical index (position), starting from 0. <br> e.g., \`my_list[0]\` | Values are accessed by their unique key. <br> e.g., \`my_dict['name']\`           |
| Mutability     | Mutable: Items can be added, removed, or changed.                | Mutable: Key-value pairs can be added, removed, or changed.                 |
| Syntax         | Defined using square brackets \`[]\`. <br> \`[10, 'hello', True]\`      | Defined using curly braces \`{}\` with colons separating keys and values. <br> \`{'name': 'Alice', 'age': 25}\` |
| Keys/Indices   | Indices are always integers (0, 1, 2, ...).                          | Keys must be of an immutable type (e.g., string, number, or tuple). Keys must be unique. |
| Use Case       | Ideal for storing a sequence of items where the order is important and you need to access elements by position. | Perfect for storing related information where each piece of data has a unique identifier (the key). It's a mapping from keys to values. |

**Summary:**
- Use a List when you have a collection of items and their order matters. Think of it as a simple sequence or an array.
- Use a Dictionary when you need to store data as logical associations, like a phone book (name is the key, number is the value) or a description of an object (attribute name is the key, attribute value is the value). It provides fast lookups based on a unique identifier.
          `,
        },
      },
       {
        id: '24-Q1C-Fib',
        question: 'OR C. Write a python program to display the Fibonacci sequence up to nth term using function.',
        answer: {
            theory: `
**Concept of the Fibonacci Sequence**

The Fibonacci sequence is a famous mathematical series where each number is the sum of the two preceding ones. The sequence typically starts with 0 and 1.
The sequence goes: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...

Where:
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

**Algorithm for an Iterative Solution**

While Fibonacci can be solved recursively, an iterative approach (using a loop) is generally more efficient as it avoids the overhead of repeated function calls and the risk of "maximum recursion depth exceeded" errors for large numbers.

The logic for the function is as follows:
1.  **Function Definition:** Create a function that accepts one argument, \`n\`, representing the number of terms to generate.
2.  **Handle Edge Cases:**
    - If \`n\` is less than or equal to 0, return an empty list as there are no terms to display.
    - If \`n\` is 1, return a list containing just the first term, \`[0]\`.
3.  **Initialization:** Create a list and initialize it with the first two Fibonacci numbers, \`[0, 1]\`.
4.  **Iteration:** Use a loop that continues as long as the number of terms in our list is less than the desired number, \`n\`.
    - Inside the loop, calculate the next term by adding the last two numbers in the sequence. In Python, these can be accessed with \`sequence[-1]\` (last element) and \`sequence[-2]\` (second to last element).
    - Append this new term to the list.
5.  **Return Value:** Once the loop finishes, the list will contain the complete sequence up to the nth term. Return this list.
            `,
            code: `def generate_fibonacci(n_terms):
    """
    This function generates the Fibonacci sequence up to a specified number of terms
    using an iterative approach.
    
    Args:
        n_terms (int): The number of terms to generate. Must be a positive integer.

    Returns:
        list: A list containing the Fibonacci sequence.
    """
    # Step 2: Handle edge cases
    if n_terms <= 0:
        return []
    elif n_terms == 1:
        return [0]
    
    # Step 3: Initialize the sequence with the first two terms
    sequence = [0, 1]
    
    # Step 4: Generate the rest of the terms using a loop
    # The loop runs until the list has the desired number of terms.
    while len(sequence) < n_terms:
        # Calculate the next term by summing the last two
        next_term = sequence[-1] + sequence[-2]
        # Append the new term to the sequence
        sequence.append(next_term)
        
    # Step 5: Return the complete sequence
    return sequence

# --- Main part of the program to interact with the user ---
if __name__ == "__main__":
    try:
        # Get user input and cast it to an integer
        num = int(input("Enter the number of Fibonacci terms to display: "))
        
        if num <= 0:
            print("Please enter a positive integer.")
        else:
            # Call the function to get the result
            fib_sequence = generate_fibonacci(num)
            print(f"\\nThe Fibonacci sequence with {num} terms is:")
            print(fib_sequence)

    except ValueError:
        print("Invalid input. Please enter a valid integer.")`,
            output: `Assuming the user enters '10':

Enter the number of Fibonacci terms to display: 10

The Fibonacci sequence with 10 terms is:
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]`
        }
       },
       {
        id: '24-Q3C-Loan',
        question: `C. Assume we have csv file of Loan dataset and attributes of the datasets are – Loan_id, Gender, Brand, Marital_status, Education, Income, Loan_amont, Loan_term. Perform following operation using panda’s dataframe and write python code snippet for it.
1) Read the csv file using panda’s dataframe object
2) Describe and display statistical details of dataset using pandas.
3) Remove the column Loan_term and display the information about size of the dataset.
4) Sort the dataset according to the income of the applicant and display the first five records with highest income.`,
        answer: {
            theory: `
**Explanation of Concepts and Pandas Functions**

This problem involves performing common data manipulation and analysis tasks using the Pandas library, which is the cornerstone of data science in Python.

**1. Reading a CSV File:**
   - \`pandas.read_csv()\`: This is the primary function for loading data from a comma-separated values (CSV) file into a Pandas DataFrame. A DataFrame is a 2D labeled data structure, similar to a spreadsheet, that is optimized for data analysis. For this example, we simulate a file using \`io.StringIO\` to make the code self-contained and reproducible.

**2. Describing Statistical Details:**
   - \`DataFrame.describe()\`: This method is a powerful tool for getting a quick statistical summary of the numerical columns in a DataFrame. The output includes:
     - count: The number of non-missing entries.
     - mean: The average value.
     - std: The standard deviation, which measures the amount of variation or dispersion.
     - min: The minimum value.
     - 25% (Q1): The first quartile. 25% of the data is below this value.
     - 50% (Q2): The median. 50% of the data is below this value.
     - 75% (Q3): The third quartile. 75% of the data is below this value.
     - max: The maximum value.

**3. Removing a Column and Checking Size:**
   - \`DataFrame.drop(columns=[...])\`: This method is used to remove specified rows or columns. By using the \`columns\` parameter, we tell it to drop the 'Loan_term' column.
   - \`DataFrame.shape\`: This attribute returns a tuple representing the dimensions of the DataFrame in the format \`(rows, columns)\`. It's a quick way to check the size of the dataset after a modification.

**4. Sorting and Displaying Records:**
   - \`DataFrame.sort_values()\`: This method sorts the DataFrame by the values in one or more columns.
     - \`by='Income'\`: Specifies that the sorting should be based on the 'Income' column.
     - \`ascending=False\`: Sorts the data in descending order (highest to lowest).
   - \`DataFrame.head(n)\`: This method returns the first \`n\` rows of the DataFrame. It's commonly used after sorting to view the top records, such as the applicants with the highest incomes.
            `,
            code: `import pandas as pd
from io import StringIO

# To make this example runnable, we first create a dummy CSV data string
# that simulates the content of the 'Loan.csv' file.
csv_data = """Loan_id,Gender,Brand,Marital_status,Education,Income,Loan_amount,Loan_term
LP001,Male,Apple,Yes,Graduate,5849,128,360
LP002,Female,Samsung,No,Graduate,4583,128,360
LP003,Male,OnePlus,Yes,Not Graduate,3000,66,360
LP004,Male,Apple,Yes,Graduate,6000,141,360
LP005,Female,Xiaomi,No,Graduate,10000,200,180
LP006,Male,Google,Yes,Graduate,2583,123,360
LP007,Male,Samsung,No,Not Graduate,9500,180,360
"""

# 1) Read the csv file into a pandas DataFrame object
# We use StringIO to treat the string 'csv_data' as a file.
df = pd.read_csv(StringIO(csv_data))
print("--- 1. Initial DataFrame ---")
print(df)

# 2) Describe and display statistical details of the dataset
print("\\n--- 2. Statistical Details of the Dataset ---")
print(df.describe())

# 3) Remove the column 'Loan_term' and display the size
# The drop() method returns a new DataFrame with the column removed.
df_dropped = df.drop(columns=['Loan_term'])
print("\\n--- 3. After Removing 'Loan_term' Column ---")
print("Remaining columns:", df_dropped.columns.tolist())
# The .shape attribute gives a (rows, columns) tuple.
print("Size of the new dataset (rows, columns):", df_dropped.shape)


# 4) Sort by income and display the top five records
# We sort by the 'Income' column in descending order.
sorted_df = df_dropped.sort_values(by='Income', ascending=False)
print("\\n--- 4. Top 5 Records with Highest Income ---")
# .head(5) selects the first 5 rows of the sorted DataFrame.
print(sorted_df.head(5))`,
            output: `--- 1. Initial DataFrame ---
  Loan_id  Gender    Brand Marital_status     Education  Income  Loan_amount  Loan_term
0   LP001    Male    Apple            Yes      Graduate    5849          128        360
1   LP002  Female  Samsung             No      Graduate    4583          128        360
2   LP003    Male  OnePlus            Yes  Not Graduate    3000           66        360
3   LP004    Male    Apple            Yes      Graduate    6000          141        360
4   LP005  Female   Xiaomi             No      Graduate   10000          200        180
5   LP006    Male   Google            Yes      Graduate    2583          123        360
6   LP007    Male  Samsung             No  Not Graduate    9500          180        360

--- 2. Statistical Details of the Dataset ---
             Income  Loan_amount   Loan_term
count      7.000000     7.000000    7.000000
mean    5930.714286   138.000000  334.285714
std     2932.162781    45.869380   64.285714
min     2583.000000    66.000000  180.000000
25%     3791.500000   125.500000  360.000000
50%     5849.000000   128.000000  360.000000
75%     7750.000000   160.500000  360.000000
max    10000.000000   200.000000  360.000000

--- 3. After Removing 'Loan_term' Column ---
Remaining columns: ['Loan_id', 'Gender', 'Brand', 'Marital_status', 'Education', 'Income', 'Loan_amount']
Size of the new dataset (rows, columns): (7, 7)

--- 4. Top 5 Records with Highest Income ---
  Loan_id  Gender    Brand Marital_status     Education  Income  Loan_amount
4   LP005  Female   Xiaomi             No      Graduate   10000          200
6   LP007    Male  Samsung             No  Not Graduate    9500          180
3   LP004    Male    Apple            Yes      Graduate    6000          141
0   LP001    Male    Apple            Yes      Graduate    5849          128
1   LP002  Female  Samsung             No      Graduate    4583          128`
        }
       },
    ],
  },
];