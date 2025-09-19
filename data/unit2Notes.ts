import React from 'react';
import { Unit } from '../types';
import { DataScientistVennDiagram, DataEcosystemDiagram, DataSciencePipelineDiagram, TrainTestSplitDiagram } from '../components/Diagrams';

export const unit2Notes: Unit = {
    id: 2,
    title: "Unit 2: Data Science and Python",
    topics: [
        {
            title: "2.1.1 Emergence of Data Science",
            content: `The field of data science emerged as a distinct discipline in the early 21st century, driven by a "perfect storm" of technological and business trends that created both the need and the capability to analyze data at an unprecedented scale.

- **Data Explosion (The "3 V's" of Big Data):** The rise of the internet, social media, e-commerce, and IoT sensors began generating data of immense **Volume** (terabytes and petabytes), at high **Velocity** (streaming data), and in a wide **Variety** (structured tables, unstructured text, images, videos). Traditional business intelligence tools, designed for structured data in relational databases, were simply not equipped to handle this new reality.

- **Exponential Growth in Computational Power & Storage:** Moore's Law, which observed the doubling of transistors on a chip roughly every two years, led to massive increases in processing power. Simultaneously, the cost of data storage (hard drives, cloud storage) plummeted. The advent of cloud computing platforms like AWS, Google Cloud, and Azure democratized access to supercomputing power, making it affordable for startups and researchers to analyze massive datasets without owning expensive hardware.

- **Algorithmic Advancements:** Parallel to hardware improvements, sophisticated new algorithms were developed and refined. Breakthroughs in machine learning and the resurgence of deep learning (neural networks) provided the methods needed to extract complex patterns, make predictions, and find insights from messy, high-dimensional data.

- **Shift in Business Value:** Businesses began to recognize that data was a critical strategic asset. The focus shifted from retrospective reporting (what happened last quarter?) to predictive and prescriptive analytics (what will happen next, and what should we do about it?). This created a demand for a new type of professional who could combine programming skills, statistical knowledge, and business acumen to unlock the value hidden in the data.`
        },
        {
            title: "2.1.2 Core Competencies of a Data Scientist",
            content: `A data scientist is a multidisciplinary expert who blends skills from several fields. The role is often visualized as a Venn diagram of three key areas, and proficiency is needed at their intersection.

- **Computer Science & Hacking Skills:** This is the practical ability to acquire, manage, and work with data. It goes beyond just writing code.
    - **Programming:** Strong proficiency in languages like Python or R for scripting, data manipulation, and modeling.
    - **Databases:** Ability to query and extract data from both SQL (structured) and NoSQL (unstructured) databases.
    - **Big Data Technologies:** Familiarity with distributed computing frameworks like Apache Spark for handling data too large for a single machine.
    - **"Hacking" Mindset:** This refers to the creative, problem-solving ability to handle messy, incomplete, real-world data and find clever ways to make different tools and systems work together.

- **Math & Statistics Knowledge:** This is the theoretical foundation that ensures analyses are rigorous and the conclusions are sound. Without this, data science becomes a blind application of algorithms.
    - **Probability & Statistics:** Essential for designing experiments (like A/B testing), performing hypothesis testing, and quantifying uncertainty in predictions.
    - **Linear Algebra:** The language of data. Concepts like vectors and matrices are fundamental to how machine learning algorithms, especially in deep learning, represent and process data.
    - **Calculus:** Crucial for understanding how models "learn" through optimization processes like gradient descent.

- **Substantive/Domain Expertise:** This is the critical business or scientific context. A data scientist must understand the industry they are working in (e.g., finance, healthcare, marketing) to:
    - **Ask the Right Questions:** Formulate questions that are both analytically tractable and strategically valuable to the business.
    - **Feature Engineering:** Use domain knowledge to create relevant features from the raw data that will improve model performance.
    - **Communicate Results:** Translate complex technical findings into a compelling story and actionable recommendations that non-technical stakeholders can understand and act upon.`,
            diagram: {
                component: React.createElement(DataScientistVennDiagram),
                caption: 'A Venn diagram showing the intersection of Computer Science, Math & Statistics, and Domain Expertise.',
            },
        },
        {
            title: "2.1.3 Linking Data Science, Big Data, and AI",
            content: `These three terms are deeply intertwined and form a cohesive, symbiotic ecosystem. They are not interchangeable but build upon each other.

- **Big Data:** Is the raw material or the fuel. It refers to the massive, complex datasets that are too large to be handled by traditional software. It is the foundational resource that presents both the challenge and the opportunity.

- **Data Science:** Is the process or the discipline. It's the entire workflow of methodologies, tools, and scientific principles used to turn the raw material of Big Data into a refined product. This product could be knowledge, an insight, a visualization, or a predictive model. It's the "how" of extracting value from data.

- **Artificial Intelligence (AI) & Machine Learning (ML):** Are the engine and the final product. A machine learning model is an artifact created through the data science process. This model, when deployed, becomes an AI application that can perform an intelligent task automatically.

**A Clear Workflow:**
1.  A company collects **Big Data** on user behavior.
2.  A **Data Scientist** applies the data science process: they clean this data, explore it to understand patterns, and then use it to build and train a **Machine Learning** model.
3.  This trained model is deployed into an application, creating an **AI product**, such as a recommendation engine that suggests new movies to users in real-time.

The relationship is cyclical: the AI product generates more data on its interactions, which feeds back into the Big Data ecosystem, allowing data scientists to further refine and improve the AI model.`,
            diagram: {
                component: React.createElement(DataEcosystemDiagram),
                caption: 'The symbiotic relationship between Big Data (raw material), Data Science (process), and AI/ML (product).',
            },
        },
        {
            title: "2.2.1 Creating the Data Science Pipeline: Preparing the Data",
            content: `This is the foundational stage, often called "data wrangling" or "data munging." It's typically the most time-consuming part of a project but is critical for success. The principle is simple: "Garbage in, garbage out." A sophisticated model cannot compensate for poor-quality data.

- **Data Acquisition:** The process of gathering data from various sources. This could be reading a CSV file from a local disk, querying a relational database using SQL, fetching data from a web API, or scraping information from web pages.

- **Data Cleaning (Wrangling):** This involves identifying and fixing errors, inconsistencies, and missing information in the raw data.
    - **Handling Missing Values:** Data is rarely complete. You must decide whether to remove rows with missing data (\`dropna()\`) or fill them in (imputation) using statistical measures like the mean, median, or mode (\`fillna()\`).
    - **Correcting Data Types:** Ensuring columns are stored in the correct format (e.g., converting a column of dates stored as strings into proper datetime objects).
    - **Removing Duplicates:** Identifying and deleting redundant records which can skew statistical analyses and model training.
    - **Standardizing Data:** Fixing inconsistent entries, such as variations in capitalization ('USA', 'U.s.a.', 'usa') or formatting.

- **Data Transformation:** Modifying and enriching the data to make it suitable for modeling.
    - **Feature Engineering:** The creative process of constructing new, more informative features from existing ones. For example, from a timestamp, you could engineer features like 'day_of_week' or 'is_holiday'.
    - **Feature Scaling:** Normalizing or standardizing numerical data to bring them to a common scale. This is essential for many ML algorithms (like SVMs or PCA) that are sensitive to the magnitude of features.
    - **Encoding Categorical Variables:** Converting text-based categories into numbers that a model can understand, typically using techniques like one-hot encoding.`,
            diagram: {
                component: React.createElement(DataSciencePipelineDiagram),
                caption: 'A flowchart illustrating the six key stages of the Data Science Pipeline, from problem definition to deployment.',
            },
        },
        {
            title: "2.2.2 Creating the Data Science Pipeline: Exploratory Data Analysis (EDA)",
            content: `EDA is the process of "getting to know your data." The goal is to investigate the dataset to discover patterns, spot anomalies, test initial hypotheses, and check assumptions, primarily through summary statistics and graphical representations. EDA is not about producing final answers but about generating better questions and developing an intuition for the data.

- **Univariate Analysis (Analyzing one variable at a time):**
    - **Numerical Variables:** Use histograms and box plots to understand the variable's distribution (e.g., is it normally distributed or skewed?), its central tendency (mean, median), and its spread (standard deviation, interquartile range). For example, a histogram of customer ages might reveal two distinct peaks, suggesting two different customer segments.
    - **Categorical Variables:** Use bar charts and count plots to see the frequency of each category. This can help identify imbalanced classes, where one category is far more common than others, which can be problematic for model training.

- **Bivariate Analysis (Analyzing the relationship between two variables):**
    - **Numerical vs. Numerical:** Use scatter plots to visualize the relationship. Look for correlations (positive, negative, or no correlation) and patterns (linear or non-linear). For instance, a scatter plot might show a strong positive linear relationship between advertising spend and sales.
    - **Categorical vs. Numerical:** Use box plots or violin plots to compare the distribution of a numerical variable across different categories. For example, you could compare the distribution of salaries across different job departments.
    - **Categorical vs. Categorical:** Use cross-tabulations (contingency tables) or heatmaps to understand the frequency of co-occurrences.

EDA is an iterative cycle of questioning, visualizing, and summarizing that forms the foundation for good modeling.`
        },
        {
            title: "2.2.3 Creating the Data Science Pipeline: Learning from Data (Modeling)",
            content: `This is where machine learning algorithms are used to find patterns in the prepared data and build a predictive or descriptive model.

- **Model Selection:** Choosing an appropriate algorithm based on the problem type.
    - **Supervised Learning:** The data is labeled with the correct outcome. The goal is to learn a mapping function that can predict the outcome for new, unseen data. It's like teaching a student with flashcards that have both a question and the answer.
        - **Regression:** Predicting a continuous value (e.g., predicting a house price, stock price, or temperature). Common algorithms: Linear Regression, Decision Tree Regressor, Gradient Boosting.
        - **Classification:** Predicting a discrete category (e.g., predicting if an email is spam or not spam, or if a tumor is benign or malignant). Common algorithms: Logistic Regression, Support Vector Machines (SVM), Random Forest.
    - **Unsupervised Learning:** The data is unlabeled. The goal is to find hidden structure or patterns within the data itself, without a predefined outcome to predict.
        - **Clustering:** Grouping similar data points together (e.g., segmenting customers into different personas based on their purchasing behavior). Common algorithm: K-Means.
        - **Dimensionality Reduction:** Reducing the number of variables in a dataset while preserving its important structure, often for visualization or to improve performance. Common algorithm: Principal Component Analysis (PCA).

- **Training, Validation, and Testing:** To build a reliable model, the dataset is typically split into three parts:
    - **Training Set:** The model learns the patterns from this (largest) portion of the data.
    - **Validation Set:** Used to tune the model's hyperparameters (its settings) and select the best-performing model without "peeking" at the test set.
    - **Testing Set:** The final, chosen model's performance is evaluated on this completely unseen data to assess how well it will generalize to new, real-world situations. This process helps prevent overfitting, a common pitfall where a model learns the training data too well (including its noise) but fails to perform on new data.`,
            diagram: {
                component: React.createElement(TrainTestSplitDiagram),
                caption: 'Visual representation of splitting a full dataset into a larger training set and a smaller, unseen testing set.',
            },
        },
        {
            title: "2.2.4-2.2.5 Creating the Data Science Pipeline: Visualization and Creating Data Products",
            content: `The final stages of the pipeline are about communicating findings and delivering value.

- **Visualization:** This is a continuous activity, but its purpose shifts at the end of the project.
    - **Exploratory Visualization (during EDA):** Created for the data scientist's own understanding. These plots can be complex, dense, and interactive, designed to uncover hidden patterns. A pair plot showing relationships between all variables is a classic example.
    - **Explanatory Visualization (for communication):** Created to tell a compelling story to stakeholders. These plots must be clear, simple, and carefully annotated to highlight a key insight. The goal is to convey a conclusion, not just show data. A single, well-labeled bar chart showing the impact of a marketing campaign is more effective here than a complex scatter plot.

- **Obtaining Insights and Creating Data Products:** The outcome of a data science project can take several forms:
    - **An Insight:** A presentation or report detailing a key finding that informs a strategic business decision. For example, "We discovered that customers who use feature X are 50% less likely to churn, so the business should invest in promoting it more." The value is in the knowledge provided.
    - **A Data Product:** A system that integrates the trained model into a production environment to provide ongoing, automated value. The model is the core of the product. Examples include:
        - A recommendation engine on an e-commerce site.
        - A real-time fraud detection system for financial transactions.
        - A spam filter in an email client.
        - A dynamic pricing model for an airline.`
        },
        {
            title: "2.4 Python’s Capabilities and Philosophy",
            content: `- **2.4.1 Why Python?** Python has become the lingua franca of data science due to a powerful combination of being simple, versatile, and supported by an incredible open-source community. It is often called a "glue language" because it can easily connect disparate systems, making it perfect for building end-to-end data science projects, from data extraction and analysis to deploying a model in a web application.

- **2.4.2 Python’s Core Philosophy (The Zen of Python):** You can see these 19 guiding principles by typing \`import this\` into a Python interpreter. Key ideas that are highly relevant to data science include:
    - "Beautiful is better than ugly."
    - "Simple is better than complex."
    - "Readability counts."
    This philosophy encourages writing clean, understandable, and maintainable code. In data science, where projects are collaborative and experiments need to be reproducible, readable code is not a luxury—it's a necessity. It ensures that you and others can understand and verify your analysis months later.

- **2.4.6 Rapid Prototyping and Experimentation:** Python's simple syntax and the high-level nature of its libraries, especially when used in an interactive environment like a Jupyter Notebook, allow data scientists to quickly test ideas, build prototype models, and iterate on their work very efficiently. You can go from an idea to a result in minutes, not hours.

- **2.4.7 Speed of Execution:** A common misconception is that Python is "slow." While Python's interpreter is slower than compiled languages like C or Fortran, this is largely irrelevant for data science. The core data science libraries (NumPy, pandas, Scikit-learn) are high-performance wrappers around code written in C and Fortran. This gives you the best of both worlds: the ease and flexibility of writing Python code, with the near-native speed of compiled code for the heavy numerical computations that dominate data analysis. Your Python code acts as the high-level command, and the low-level, optimized C code does the actual work.`
        },
        {
            title: "2.5.1 NumPy (Numerical Python)",
            content: `NumPy is the fundamental package for scientific computing in Python. It is the bedrock upon which nearly all other data science libraries (including pandas and scikit-learn) are built.

- **Core Data Structure: The \`ndarray\`:** Its main object is the powerful N-dimensional array (\`ndarray\`). This is a grid of values, all of the same type, and is indexed by a tuple of non-negative integers. Key attributes include:
    - \`ndarray.ndim\`: the number of axes (dimensions) of the array.
    - \`ndarray.shape\`: a tuple of integers indicating the size of the array in each dimension.
    - \`ndarray.dtype\`: an object describing the type of the elements in the array (e.g., \`int64\`, \`float64\`).

- **Key Feature: Vectorization:** This is NumPy's most important feature. It allows you to perform complex mathematical operations on entire arrays at once without writing slow, explicit \`for\` loops in Python. These operations are executed in pre-compiled C code, making them orders of magnitude faster.

- **Broadcasting:** A powerful mechanism that allows NumPy to perform operations on arrays of different (but compatible) shapes. For example, you can add a single number (a scalar) to every element of a larger array.`,
            code: `import numpy as np

# Create a 2D NumPy array (a matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
print(f"Matrix:\\n{matrix}")
print(f"Shape: {matrix.shape}")
print(f"Dimensions: {matrix.ndim}")
print(f"Data type: {matrix.dtype}")

# Vectorization: Multiply every element by 10 without a loop
vectorized_result = matrix * 10
print(f"\\nVectorized Multiplication:\\n{vectorized_result}")

# Broadcasting: Add a 1D array to each row of the 2D array
vector_to_add = np.array([100, 200, 300])
broadcasted_result = matrix + vector_to_add
print(f"\\nBroadcasted Addition:\\n{broadcasted_result}")`,
            output: `Matrix:
[[1. 2. 3.]
 [4. 5. 6.]]
Shape: (2, 3)
Dimensions: 2
Data type: float64

Vectorized Multiplication:
[[10. 20. 30.]
 [40. 50. 60.]]

Broadcasted Addition:
[[101. 202. 303.]
 [104. 205. 306.]]`
        },
        {
            title: "2.5.2 Pandas",
            content: `Pandas is the essential library for data manipulation and analysis in Python. It provides high-performance, easy-to-use data structures and a rich set of data analysis tools, making it the go-to tool for real-world data wrangling and exploration.

- **Core Data Structures:**
    - **Series:** A one-dimensional labeled array, capable of holding any data type (integers, strings, floats, Python objects, etc.). It's like a single column of a spreadsheet. The main components are the data and the associated labels, which are collectively known as the index.
    - **DataFrame:** A two-dimensional labeled data structure with columns of potentially different types. It is the primary pandas data structure, analogous to a spreadsheet, a SQL table, or a dictionary of Series objects. You can think of a DataFrame as a collection of Series that share the same index.

- **Key Features:**
    - **Data I/O:** Effortlessly reads and writes data from/to a wide array of formats like CSV, Excel, SQL databases, JSON, and more.
    - **Data Wrangling:** A powerful set of tools for cleaning, filtering, grouping, merging, pivoting, and reshaping data.
    - **Powerful Indexing:** Flexible methods for selecting and filtering subsets of data using labels (\`.loc\`) and integer positions (\`.iloc\`).
    - **Time Series Functionality:** Robust tools for working with time-series data, including date range generation and frequency conversion.`,
            code: `import pandas as pd

# Create a sample DataFrame
data = {'Department': ['Sales', 'IT', 'Sales', 'IT', 'Marketing'],
        'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Salary': [70000, 85000, 72000, 90000, 65000]}
df = pd.DataFrame(data)
print("--- Original DataFrame ---")
print(df)

# --- Common Pandas Operations ---
# 1. Filtering with boolean indexing
high_earners = df[df['Salary'] > 80000]
print("\\n--- High Earners (Salary > 80k) ---")
print(high_earners)

# 2. GroupBy: Calculate multiple aggregations per department
dept_summary = df.groupby('Department')['Salary'].agg(['mean', 'max', 'count'])
print("\\n--- Department Salary Summary ---")
print(dept_summary)

# 3. Create a new column based on an existing one
df['Salary_in_Thousands'] = df['Salary'] / 1000
print("\\n--- DataFrame with New Column ---")
print(df)`,
            output: `--- Original DataFrame ---
  Department Employee  Salary
0      Sales    Alice   70000
1         IT      Bob   85000
2      Sales  Charlie   72000
3         IT    David   90000
4  Marketing      Eve   65000

--- High Earners (Salary > 80k) ---
  Department Employee  Salary
1         IT      Bob   85000
3         IT    David   90000

--- Department Salary Summary ---
                  mean    max  count
Department                           
IT           87500.0  90000      2
Marketing    65000.0  65000      1
Sales        71000.0  72000      2

--- DataFrame with New Column ---
  Department Employee  Salary  Salary_in_Thousands
0      Sales    Alice   70000                 70.0
1         IT      Bob   85000                 85.0
2      Sales  Charlie   72000                 72.0
3         IT    David   90000                 90.0
4  Marketing      Eve   65000                 65.0`
        },
        {
            title: "2.5.3 Matplotlib",
            content: `Matplotlib is the original and most widely used plotting library for Python. It provides a huge degree of control over every aspect of a figure, from line styles and colors to axis labels and titles. While it can be verbose for complex plots, its flexibility is unparalleled. It is the foundation upon which other higher-level libraries, like Seaborn, are built.

- **Key Concepts:**
    - **Figure:** The top-level container for all the plot elements (the entire window or page on which everything is drawn).
    - **Axes:** The actual plot itself (the area with the x-axis, y-axis, etc.). A figure can contain one or more axes (subplots), which allows for creating grids of plots.

- **Usage Styles:**
    - **Pyplot API:** A simple, state-based interface that implicitly tracks the "current" figure and axes (e.g., \`plt.plot()\`, \`plt.title()\`). It's quick for simple, one-off plots but can become confusing when managing multiple plots.
    - **Object-Oriented (OO) API:** The more powerful and flexible style, which is the best practice for any non-trivial plot. You explicitly create Figure and Axes objects and call methods on them (e.g., \`fig, ax = plt.subplots()\`, \`ax.plot()\`, \`ax.set_title()\`). This makes the code more explicit, reusable, and easier to manage for complex visualizations.`,
            code: `import matplotlib.pyplot as plt
import numpy as np

# Create some data for plotting
x = np.linspace(0, 10, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Use the Object-Oriented API (preferred)
# Create a figure containing two subplots arranged vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Plot data on the first axes
ax1.plot(x, y_sin, color='blue', label='sin(x)')
ax1.set_title("Sine and Cosine Functions")
ax1.set_ylabel("Sine Value")
ax1.grid(True)
ax1.legend()

# Plot data on the second axes
ax2.plot(x, y_cos, color='red', label='cos(x)')
ax2.set_ylabel("Cosine Value")
ax2.set_xlabel("X value")
ax2.grid(True)
ax2.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()`
        },
        {
            title: "2.5.4 Scikit-learn",
            content: `Scikit-learn is the premier, all-in-one library for traditional machine learning in Python. It is built on NumPy, SciPy, and Matplotlib and provides a simple, consistent, and elegant interface to a vast array of ML algorithms. Its focus is on ease of use, performance, and solid documentation.

- **Key Features:**
    - **Comprehensive Algorithms:** Includes a vast array of supervised and unsupervised learning algorithms for classification, regression, clustering, and dimensionality reduction.
    - **Consistent Estimator API:** This is scikit-learn's most brilliant design feature. All "estimator" objects (models) share a simple, consistent interface, making it incredibly easy to swap between different models. The core methods are:
        - \`estimator.fit(X, y)\`: to train (or "fit") the model on the training data (\`X\`) and target labels (\`y\`).
        - \`estimator.predict(X_new)\`: to make predictions on new, unseen data.
        - \`estimator.transform(X)\`: for preprocessing objects that clean or modify data (e.g., scalers).
    - **Data Preparation and Model Evaluation:** Provides essential tools for every step of the modeling workflow, including splitting data (\`train_test_split\`), preprocessing (\`StandardScaler\`), and model evaluation metrics (\`accuracy_score\`, \`mean_squared_error\`).`,
            code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # Import a different classifier
from sklearn.metrics import accuracy_score

# 1. Load sample data
# X contains the features (sepal length/width, petal length/width)
# y contains the target label (the species of iris)
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data into training and testing sets
# random_state ensures the split is the same every time, for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create a model instance (K-Nearest Neighbors)
# The API is the same as for LogisticRegression
model = KNeighborsClassifier(n_neighbors=3)

# 4. Train (fit) the model on the training data
model.fit(X_train, y_train)

# 5. Make predictions on the unseen test data
predictions = model.predict(X_test)

# 6. Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")`,
            output: 'Model Accuracy: 1.0000'
        },
        {
            title: "2.5.5 SciPy (Scientific Python)",
            content: `SciPy is a library that builds on NumPy and provides a large collection of algorithms and functions for more specialized scientific and technical computing tasks. While NumPy provides the basic array structure and fundamental operations, SciPy provides the higher-level functions to perform complex analysis on those arrays.

- **Key Modules:**
    - \`scipy.stats\`: Contains a large number of probability distributions and a growing library of statistical functions. This is heavily used for hypothesis testing (e.g., t-tests, ANOVA).
    - \`scipy.optimize\`: For solving optimization problems, like finding the minimum of a function or performing curve fitting.
    - \`scipy.linalg\`: Contains more advanced linear algebra routines than NumPy, such as matrix decompositions.
    - \`scipy.integrate\`: For numerical integration and solving ordinary differential equations (ODEs).
    - \`scipy.signal\`: For signal processing tasks.`,
            code: `from scipy import stats
import numpy as np

# Imagine we have two sets of test scores from two different teaching methods
# We want to know if there is a statistically significant difference between them.
group_a_scores = np.random.normal(loc=80, scale=5, size=30)
group_b_scores = np.random.normal(loc=83, scale=5, size=30)

# Use a two-sample t-test from scipy.stats
# The null hypothesis is that the two groups have the same mean.
t_statistic, p_value = stats.ttest_ind(group_a_scores, group_b_scores)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05 # Significance level
if p_value < alpha:
    print("The difference is statistically significant (reject the null hypothesis).")
else:
    print("The difference is not statistically significant (fail to reject the null hypothesis).")`,
            output: `T-statistic: -2.3912
P-value: 0.0202
The difference is statistically significant (reject the null hypothesis).`
        },
        {
            title: "2.5.6 Deep Learning: Keras and TensorFlow",
            content: `- **TensorFlow:** A powerful, open-source platform developed by Google for large-scale machine learning and, especially, deep learning. It's a low-level library that provides immense flexibility for building and training complex neural networks by defining and running computations on dataflow graphs. It provides fine-grained control over every aspect of a model, which is essential for research and complex, novel architectures.

- **Keras:** A very user-friendly, high-level neural networks API that runs on top of TensorFlow. It is designed for fast experimentation and allows you to build complex neural network models in a modular way, by stacking layers together like Lego bricks. For most common deep learning tasks (image classification, text generation), Keras provides a much simpler and more intuitive interface than using raw TensorFlow.

**Analogy:** TensorFlow is like a full car engine with all its components exposed; you can build anything, but it requires deep mechanical knowledge. Keras is like the car's dashboard, steering wheel, and pedals; it provides a simple, intuitive interface to control the powerful engine underneath. Since Keras was integrated into TensorFlow 2.0, it is now the standard, recommended way to build models for most users.`,
            code: `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Example of building a simple Sequential model in Keras for image classification
model = keras.Sequential([
    # Input layer: flattens a 28x28 pixel image into a 1D array of 784 pixels
    layers.Flatten(input_shape=(28, 28)),
    # Hidden layer with 128 neurons and ReLU activation function
    layers.Dense(128, activation='relu'),
    # Dropout layer to prevent overfitting
    layers.Dropout(0.2),
    # Output layer with 10 neurons (for 10 classes, e.g., digits 0-9)
    # Softmax activation converts outputs to probabilities
    layers.Dense(10, activation='softmax')
])

# Compile the model, specifying the optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model's architecture
model.summary()`,
            output: `Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense (Dense)               (None, 128)               100480    
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________`
        },
        {
            title: "2.5.7 Other Key Libraries",
            content: `- **Seaborn:** A data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. Use Seaborn when you want to create complex statistical plots (like heatmaps, violin plots, or pair plots) directly from a pandas DataFrame with very little code and with aesthetically pleasing defaults.

- **Beautiful Soup:** A library for web scraping. It excels at parsing HTML and XML documents. Use Beautiful Soup when the data you need is embedded in a website's HTML, but there's no official CSV download or API available. It helps you navigate the HTML structure to extract the specific pieces of information you need.

- **NetworkX:** A library for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks (graphs). A graph is a data structure consisting of nodes (or vertices) and edges (the connections between them). Use NetworkX when your data is about relationships, such as analyzing a social network, modeling flight connections between airports, or understanding interactions between proteins.`
        }
    ]
};