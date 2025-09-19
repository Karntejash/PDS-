import { FlashcardDeck } from '../types';

export const flashcardDecks: FlashcardDeck[] = [
    {
        id: 'core-concepts',
        title: 'Core Python Concepts',
        description: 'Fundamental definitions, data structure comparisons, and syntax patterns frequently asked in exams.',
        cards: [
            { id: 1, question: "What is Python's 'Dynamic Typing'?", answer: "Variable types are determined at runtime, not declared in advance. A variable can hold an integer, then a string." },
            { id: 2, question: "What is the key difference between a List and a Tuple?", answer: "Mutability. Lists are **mutable** (can be changed), while Tuples are **immutable** (cannot be changed)." },
            { id: 3, question: "When to use a Set vs. a List?", answer: "Use a **Set** for storing **unique** items and for very **fast membership testing** (`in` keyword). Use a **List** when order matters and you need to store duplicates." },
            { id: 4, question: "What is a Dictionary?", answer: "An **unordered** (ordered in Python 3.7+) collection of **key-value pairs**. Keys must be unique and immutable. Excellent for fast lookups by key." },
            { id: 5, question: "How does the `input()` function work and why is type casting needed?", answer: "The `input()` function **always** returns a **string**. You must use functions like `int()` or `float()` to cast the string to a number for calculations." },
            { id: 6, question: "What is List Comprehension?", answer: "A concise, readable way to create lists. Syntax: `[expression for item in iterable if condition]`." },
            { id: 7, question: "How do you sort a list of dictionaries by a specific key?", answer: "Use the `sorted()` function with a `lambda` function for the `key` argument: `sorted(my_list, key=lambda x: x['price'])`." },
            { id: 8, question: "What is the difference between `==` and `is`?", answer: "`==` checks for **equality** of values. `is` checks for **identity** (if two variables point to the exact same object in memory)." },
        ]
    },
    {
        id: 'pandas-numpy',
        title: 'Pandas & NumPy Essentials',
        description: 'A rapid review of the most critical functions for data manipulation, cleaning, and analysis.',
        cards: [
            { id: 9, question: "What is the primary data structure in Pandas?", answer: "The **DataFrame**, a 2-dimensional labeled data structure with columns of potentially different types, similar to a spreadsheet." },
            { id: 10, question: "What is the primary data structure in NumPy?", answer: "The **ndarray** (N-dimensional array), which is a grid of values of the same type. It enables fast vectorized operations." },
            { id: 11, question: "What function is used to read a CSV file into a DataFrame?", answer: "`pd.read_csv('file_path.csv')`" },
            { id: 12, question: "How do you view the first 5 rows of a DataFrame `df`?", answer: "`df.head()`" },
            { id: 13, question: "How do you get a statistical summary of the numerical columns in `df`?", answer: "`df.describe()`" },
            { id: 14, question: "How do you check for missing values in `df`?", answer: "`df.isnull().sum()`" },
            { id: 15, question: "How do you remove all rows with any missing values from `df`?", answer: "`df.dropna()`" },
            { id: 16, question: "How do you fill all missing values in `df` with 0?", answer: "`df.fillna(0)`" },
            { id: 17, question: "How do you select a single column named 'age' from `df`?", answer: "`df['age']` (returns a Series)" },
            { id: 18, question: "How do you select rows where 'age' is greater than 30?", answer: "`df[df['age'] > 30]` (Boolean Indexing)" },
            { id: 19, question: "What is the difference between `.loc` and `.iloc`?", answer: "`.loc` selects by **label/index name** (inclusive). `.iloc` selects by **integer position** (exclusive)." },
            { id: 20, question: "What is the purpose of `df.groupby('column')`?", answer: "To group the DataFrame by categories in a column, allowing you to apply an aggregation function (like `.mean()`, `.sum()`) to each group." },
        ]
    },
    {
        id: 'ds-ml-workflow',
        title: 'Data Science & ML Workflow',
        description: 'Key stages of a data science project and the core Scikit-learn API. Perfect for a big-picture overview.',
        cards: [
            { id: 21, question: "What are the main stages of the Data Science Pipeline?", answer: "1. Problem Definition & Data Acquisition\n2. Data Preparation & Cleaning\n3. Exploratory Data Analysis (EDA)\n4. Modeling\n5. Visualization & Communication\n6. Deployment" },
            { id: 22, question: "What is the purpose of Exploratory Data Analysis (EDA)?", answer: "To investigate datasets to summarize their main characteristics, often using visualizations. The goal is to find patterns, spot anomalies, and form hypotheses." },
            { id: 23, question: "Why do we split data into training and testing sets?", answer: "To evaluate the model's performance on **unseen data**. This helps ensure the model **generalizes** well and hasn't just memorized the training data (overfitting)." },
            { id: 24, question: "What is the function of `model.fit(X_train, y_train)` in Scikit-learn?", answer: "To **train** the machine learning model. The model learns the patterns and relationships between the features (`X_train`) and the target labels (`y_train`)." },
            { id: 25, question: "What is the function of `model.predict(X_test)` in Scikit-learn?", answer: "To make **predictions** on new, unseen data (`X_test`) after the model has been trained." },
            { id: 26, question: "What is Matplotlib?", answer: "The foundational plotting library in Python. It provides a high degree of control for creating static, animated, and interactive visualizations." },
            { id: 27, question: "What is Seaborn?", answer: "A high-level data visualization library based on Matplotlib. It integrates well with Pandas DataFrames and is designed to create attractive statistical graphics with less code." },
            { id: 28, question: "What is Scikit-learn?", answer: "The premier machine learning library for Python. It provides a consistent API for a wide range of classification, regression, and clustering algorithms, plus tools for data preparation and model evaluation." },
        ]
    }
];