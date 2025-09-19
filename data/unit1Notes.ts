import { Unit } from '../types';

export const unit1Notes: Unit = {
    id: 1,
    title: "Unit 1: Overview of Python and Data Structures",
    topics: [
        {
            title: "1.1.1 Basics of Python: Data Types",
            content: `Python is a dynamically-typed language. This means you don't need to declare a variable's data type; the interpreter automatically determines it when you assign a value. The fundamental data types are:
- **String (str):** A sequence of Unicode characters representing text. Strings are immutable, meaning they cannot be changed after creation. Any operation that seems to modify a string (like \`.upper()\`) actually creates a new string in memory. They are defined with single (' '), double (" "), or triple (''' ''') quotes.
- **Integer (int):** Represents positive or negative whole numbers of unlimited size (e.g., 10, -300, 0).
- **Float (float):** Represents real numbers with a decimal point (e.g., 3.14, -0.01). They are implemented as double-precision floating-point numbers, which means they have finite precision and can sometimes lead to small rounding errors.
- **Boolean (bool):** Represents one of two logical states: \`True\` or \`False\`. Booleans are the cornerstone of control flow (e.g., \`if\` statements). In numeric contexts, \`True\` evaluates to 1 and \`False\` to 0.

**Type Casting:** This is the process of explicitly converting a variable from one data type to another. It's essential when, for example, you get user input (which is always a string) but need to perform mathematical calculations. This is done using constructor functions like \`int()\`, \`float()\`, and \`str()\`. If a conversion is not possible, Python will raise a \`ValueError\`.`,
            code: `course_name = "Data Science" # String
student_count = 50       # Integer
pass_rate = 95.5         # Float
is_enrolled = True       # Boolean

# The type() function reveals a variable's data type
print(f"'{course_name}' is of type: {type(course_name)}")
print(f"{student_count} is of type: {type(student_count)}")
print(f"{pass_rate} is of type: {type(pass_rate)}")
print(f"{is_enrolled} is of type: {type(is_enrolled)}")

# Type Casting: Converting user input for calculation
age_str = input("Enter your age: ") # e.g., user enters '25'
try:
    # We must cast the string to an integer for math
    age_int = int(age_str) 
    print(f"In ten years, you will be {age_int + 10} years old.")
except ValueError:
    print(f"Error: Could not convert '{age_str}' to an integer.")`,
            output: `'Data Science' is of type: <class 'str'>
50 is of type: <class 'int'>
95.5 is of type: <class 'float'>
True is of type: <class 'bool'>
Enter your age: 25
In ten years, you will be 35 years old.`
        },
        {
            title: "1.1.2 Basics of Python: Variables",
            content: `A variable in Python is a symbolic name that acts as a reference or a pointer to an object in memory. When you assign a value to a variable, you are binding the name to that object.
- **Assignment:** The equals sign (\`=\`) is the assignment operator.
- **Dynamic Typing:** You can reassign a variable to an object of a different type at any time. Python doesn't associate a data type with the variable name, but with the object it points to.
- **Naming Conventions (PEP 8 Style Guide):**
    - **snake_case:** Variable names should be lowercase, with words separated by underscores (e.g., \`first_name\`). This improves readability.
    - Names must start with a letter (a-z, A-Z) or an underscore (_).
    - Names are case-sensitive (\`age\`, \`Age\`, and \`AGE\` are three different variables).
    - Avoid using Python keywords (like \`if\`, \`for\`, \`class\`) as variable names.`,
            code: `# Assigning values to variables following snake_case convention
first_name = "Alex"
user_age = 30
print(f"Name: {first_name}, Age: {user_age}")

# Re-assignment demonstrates dynamic typing
user_age = "thirty years old"
print(f"\\nThe variable 'user_age' now points to a string.")
print(f"New value: {user_age}, New type: {type(user_age)}")

# Multiple variables can reference the same mutable object
my_list_a = [1, 2, 3]
my_list_b = my_list_a # Both variables point to the SAME list in memory
my_list_b.append(4)

print(f"\\nList A: {my_list_a}") # The change is reflected here too!
print(f"List B: {my_list_b}")`,
            output: `Name: Alex, Age: 30

The variable 'user_age' now points to a string.
New value: thirty years old, New type: <class 'str'>

List A: [1, 2, 3, 4]
List B: [1, 2, 3, 4]`
        },
        {
            title: "1.1.3 Basics of Python: Expressions",
            content: `An expression is a combination of values, variables, operators, and function calls that the Python interpreter evaluates to produce a value.
- **Arithmetic Operators:** \`+\` (add), \`-\` (subtract), \`*\` (multiply), \`/\` (float division), \`//\` (integer/floor division), \`%\` (modulus/remainder), \`**\` (exponentiation).
- **Comparison (Relational) Operators:** Compare two values and return a boolean (\`True\` or \`False\`). Includes \`==\` (equal), \`!=\` (not equal), \`>\`, \`<\`, \`>=\`, \`<=\`.
- **Logical Operators:** Combine boolean expressions.
    - **\`and\`:** True only if both operands are true.
    - **\`or\`:** True if at least one operand is true.
    - **\`not\`:** Inverts the boolean value.
- **Operator Precedence:** Python evaluates expressions in a specific order, commonly remembered by PEMDAS (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction). Use parentheses \`()\` to explicitly control the order of evaluation and improve readability.`,
            code: `# Arithmetic expression
# Without parentheses, 2*4 is calculated first due to precedence.
result1 = 20 - 5 * 2 # 20 - 10 = 10
# With parentheses, (20-5) is calculated first.
result2 = (20 - 5) * 2 # 15 * 2 = 30
print(f"Without parentheses: {result1}")
print(f"With parentheses: {result2}")

# Comparison and Logical expression
age = 25
has_license = True
has_good_credit = False

# Can this person rent a car? They must be 25+ AND have a license.
can_rent_car = age >= 25 and has_license 
print(f"Can rent car? {can_rent_car}")

# Can this person get a loan? They need good credit OR be over 30.
can_get_loan = has_good_credit or age > 30 
print(f"Can get loan? {can_get_loan}")`,
            output: `Without parentheses: 10
With parentheses: 30
Can rent car? True
Can get loan? False`
        },
        {
            title: "1.1.4 Basics of Python: Objects and Functions",
            content: `- **Objects:** In Python, everything is an object. An object is a self-contained entity that consists of data (attributes) and associated procedures (methods). For example, a string object contains the character data and also has methods like \`.upper()\` and \`.split()\` that can act on that data. You can think of an object as a noun and its methods as verbs.
- **Functions:** A function is a named, reusable block of organized code designed to perform a single, specific task. Functions improve the modularity and reusability of your code.
    - **Definition:** Use the \`def\` keyword, followed by the function name, parentheses \`()\` for parameters, and a colon \`:\`. The code block within the function must be indented.
    - **Parameters and Arguments:** A parameter is the variable listed inside the parentheses in the function definition. An argument is the actual value sent to the function when it is called.
    - **Return Value:** The \`return\` statement exits a function and passes back a value to the caller. A function with no \`return\` statement implicitly returns \`None\`.
    - **Docstrings:** The first statement of a function can be a string literal, which serves as the function's documentation string (docstring). It explains the function's purpose, arguments, and return value.`,
            code: `def calculate_area(length, width=10):
  """Calculates the area of a rectangle.
  
  Args:
    length (int or float): The length of the rectangle.
    width (int or float, optional): The width. Defaults to 10.
    
  Returns:
    int or float: The calculated area.
  """
  return length * width

# Calling with a positional argument
area1 = calculate_area(5) # Uses the default width of 10

# Calling with both positional arguments
area2 = calculate_area(5, 20)

# Calling with keyword arguments (order doesn't matter)
area3 = calculate_area(width=8, length=5)

print(f"Area 1 (default width): {area1}")
print(f"Area 2 (positional): {area2}")
print(f"Area 3 (keyword): {area3}")`,
            output: `Area 1 (default width): 50
Area 2 (positional): 100
Area 3 (keyword): 40`
        },
        {
            title: "1.2.1 Python Data Structures: String",
            content: `A string is an ordered sequence of characters. It is an essential data type for handling textual data.
- **Immutability:** A critical characteristic of strings. Once a string is created, its contents cannot be changed. Methods that appear to modify a string, like \`.replace()\`, actually return a new string with the modification.
- **Indexing & Slicing:** Access individual characters with \`[index]\` (0-based) or a range of characters with slice notation \`[start:stop:step]\`. Negative indices count from the end.
- **Common Methods:**
    - Case conversion: \`.upper()\`, \`.lower()\`.
    - Whitespace manipulation: \`.strip()\`.
    - Searching: \`.startswith(prefix)\`, \`.find(substring)\`.
    - Transformation: \`.split(delimiter)\`, \`.join(iterable)\`, \`.replace(old, new)\`.
- **f-Strings (Formatted String Literals):** The modern, preferred way to format strings. They are fast and allow you to embed expressions directly inside string literals by prefixing the string with an 'f' and placing expressions in curly braces \`{}\`.`,
            code: `raw_data = "  user_id,item_A,25.99\\n"

# Method chaining to clean the string
clean_data = raw_data.strip().upper().replace(",", "|")
print(f"Cleaned data: {clean_data}")

# Splitting the string into a list of values
data_points = clean_data.split('|')
print(f"Split into a list: {data_points}")

# Indexing and Slicing
user_id = data_points[0]
item_id = data_points[1]
print(f"User: {user_id}, Item: {item_id}")

# Negative indexing gets the last item
price = data_points[-1] 
print(f"Price: {price}")`,
            output: `Cleaned data: USER_ID|ITEM_A|25.99
Split into a list: ['USER_ID', 'ITEM_A', '25.99']
User: USER_ID, Item: ITEM_A
Price: 25.99`
        },
        {
            title: "1.2.2 Python Data Structures: Array",
            content: `Python has a built-in \`array\` module, but it is rarely used in data science. It provides a memory-efficient way to store a sequence of items of the same numerical type.
- **Why it's not used in Data Science:** The Python \`array\` object is a thin wrapper on C arrays. It lacks the vast functionality needed for data analysis. The scientific computing community has universally adopted NumPy arrays as the standard. NumPy arrays are more powerful, more flexible, and offer a massive library of high-performance mathematical functions (vectorized operations) that are essential for data science.
- **Key Characteristics:**
    - Type Constrained: All elements must be of the same specified 'type code' (e.g., 'i' for integer, 'f' for float).
    - Memory Efficient: Uses less memory than a list for storing large amounts of numerical data.
- **Conclusion:** While it's good to know that this structure exists, you should focus your efforts on learning NumPy for any serious numerical work.`,
            code: `import array
# Note: For data science, you would almost always use: import numpy as np

# Create an array of signed integers ('i')
# This is less common in data science than np.array([10, 20, 30, 40])
int_array = array.array('i', [10, 20, 30, 40])
int_array.append(50)
print(f"Python built-in array: {int_array}")
print(f"Its type code is: '{int_array.typecode}'")

# This is the NumPy equivalent, which is far more powerful
# import numpy as np
# np_array = np.array([10, 20, 30, 40, 50])`,
            output: `Python built-in array: array('i', [10, 20, 30, 40, 50])
Its type code is: 'i'`
        },
        {
            title: "1.2.3 Python Data Structures: List",
            content: `A list is an ordered and mutable (changeable) collection of items. It is one of the most versatile and fundamental data structures in Python.
- **Characteristics:**
    - **Ordered:** Items maintain a specific order. When you add new items, they are placed at the end unless specified otherwise.
    - **Mutable:** You can add, remove, or change items in a list after it has been created.
    - **Heterogeneous:** A single list can contain items of different data types (e.g., integers, strings, and even other lists).
- **Common Methods:**
    - \`.append(item)\`: Adds a single item to the end of the list.
    - \`.extend(iterable)\`: Appends all items from an iterable (like another list) to the end.
    - \`.insert(index, item)\`: Adds an item at a specified position.
    - \`.remove(item)\`: Removes the first occurrence of a specific value.
    - \`.pop(index)\`: Removes and returns the item at a specified position (or the last item if no index is provided).
- **List Comprehensions:** A powerful and concise syntax for creating lists. It provides a more readable and often faster alternative to using loops and \`.append()\`.`,
            code: `libraries = ["Numpy", "Pandas", "Matplotlib"]
print(f"Original list: {libraries}")

# Mutability: Change an item
libraries[2] = "Seaborn"
print(f"After modification: {libraries}")

# Add items
libraries.append("Scikit-learn")
libraries.extend(["Keras", "TensorFlow"])
print(f"After adding items: {libraries}")

# Remove and get the last item
last_item = libraries.pop()
print(f"Popped item: {last_item}")
print(f"List after pop: {libraries}")

# List comprehension: Get libraries with names longer than 5 chars
long_name_libs = [lib for lib in libraries if len(lib) > 5]
print(f"Long-named libraries: {long_name_libs}")`,
            output: `Original list: ['Numpy', 'Pandas', 'Matplotlib']
After modification: ['Numpy', 'Pandas', 'Seaborn']
After adding items: ['Numpy', 'Pandas', 'Seaborn', 'Scikit-learn', 'Keras', 'TensorFlow']
Popped item: TensorFlow
List after pop: ['Numpy', 'Pandas', 'Seaborn', 'Scikit-learn', 'Keras']
Long-named libraries: ['Pandas', 'Seaborn', 'Scikit-learn']`
        },
        {
            title: "1.2.4 Python Data Structures: Tuple",
            content: `A tuple is an ordered and immutable (unchangeable) collection of items. It is similar to a list, but its immutability gives it special properties.
- **Characteristics:**
    - **Ordered:** Items have a defined order.
    - **Immutable:** Once a tuple is created, you cannot add, remove, or change its items. This provides data integrity.
    - **Heterogeneous:** Can contain items of different data types.
- **Why Use a Tuple?**
    - **Data Integrity:** When you have data that should not change, a tuple protects it from accidental modification.
    - **Performance:** Tuples are slightly more memory-efficient and faster to process than lists.
    - **Dictionary Keys:** A list cannot be a dictionary key because it is mutable. A tuple, being immutable, can be used as a key.
- **Tuple Packing and Unpacking:**
    - **Packing:** When you assign comma-separated values to a variable, they are "packed" into a tuple: \`my_tuple = 1, 'a', True\`.
    - **Unpacking:** You can assign a tuple of N items to N variables, which unpacks the tuple's values into those variables: \`x, y, z = my_tuple\`. This is very common for functions that return multiple values.`,
            code: `def get_min_max(numbers):
    """Calculates min and max for a list of numbers."""
    return min(numbers), max(numbers) # Returns a packed tuple

data = [10, 50, 20, 90, 40]

# The function returns a tuple, which we unpack into two variables
min_val, max_val = get_min_max(data)
print(f"Min: {min_val}, Max: {max_val}")

# Tuples as dictionary keys (a list would cause a TypeError)
location_data = {
    (34.05, -118.24): "Los Angeles",
    (40.71, -74.00): "New York"
}
print(f"Location for (40.71, -74.00): {location_data[(40.71, -74.00)]}")

# Trying to change an item will raise a TypeError
# min_val[0] = 5 # This line would cause an error`,
            output: `Min: 10, Max: 90
Location for (40.71, -74.00): New York`
        },
        {
            title: "1.2.5 Python Data Structures: Set",
            content: `A set is an unordered, mutable collection of unique items. Its main characteristics are inspired by mathematical sets.
- **Characteristics:**
    - **Unordered:** Items have no defined order. You cannot access items using an index.
    - **Unique:** Sets automatically enforce uniqueness. If you add a duplicate item, it is simply ignored.
    - **Mutable:** You can add or remove items from a set.
- **Key Use Cases:**
    - **Removing Duplicates:** The fastest way to remove duplicates from a list is to convert it to a set and back: \`unique_list = list(set(my_list))\`.
    - **Membership Testing:** Checking if an item exists in a collection is extremely fast with sets (\`item in my_set\`). This is because sets are implemented using hash tables, providing O(1) average time complexity.
- **Set Operations:** Sets support powerful mathematical operations:
    - \`set_a.union(set_b)\` or \`set_a | set_b\`: All items from both sets.
    - \`set_a.intersection(set_b)\` or \`set_a & set_b\`: Items that exist in both sets.
    - \`set_a.difference(set_b)\` or \`set_a - set_b\`: Items in set A but not in set B.`,
            code: `required_skills = {'Python', 'SQL', 'Statistics'}
candidate_a_skills = {'Python', 'R', 'Tableau', 'SQL'}
candidate_b_skills = {'Python', 'Communication'}

# Find the skills candidate A has that are also required
matching_skills = required_skills.intersection(candidate_a_skills)
print(f"Candidate A's matching skills: {matching_skills}")

# Find the required skills candidate B is missing
missing_skills = required_skills.difference(candidate_b_skills)
print(f"Candidate B is missing: {missing_skills}")

# Remove duplicates from a list
log_events = ['login', 'view', 'click', 'logout', 'view', 'login']
unique_events = list(set(log_events))
print(f"Unique log events: {unique_events}")`,
            output: `Candidate A's matching skills: {'Python', 'SQL'}
Candidate B is missing: {'SQL', 'Statistics'}
Unique log events: ['click', 'logout', 'view', 'login']`
        },
        {
            title: "1.2.6 Python Data Structures: Dictionary",
            content: `A dictionary is a mutable collection that stores data as key-value pairs. It's one of the most important and widely used data structures in Python. Since Python 3.7, dictionaries are insertion-ordered.
- **Characteristics:**
    - **Key-Value Pairs:** Each item consists of a unique key and its associated value.
    - **Unique Keys:** Keys within a dictionary must be unique. If you assign a value to an existing key, it will overwrite the old value.
    - **Immutable Keys:** Keys must be of an immutable data type (e.g., string, number, tuple). This is because the dictionary needs to be able to hash the key to find the value quickly.
- **Common Methods:**
    - \`.get(key, default)\`: A safe way to access a value. If the key doesn't exist, it returns \`None\` (or a specified default value) instead of raising a \`KeyError\`.
    - \`.keys()\`, \`.values()\`, \`.items()\`: Return "view" objects that let you iterate over the dictionary's keys, values, or (key, value) pairs.
    - \`.update(other_dict)\`: Merges another dictionary into the current one.`,
            code: `student = { "name": "Alice", "age": 21, "major": "Data Science" }

# Safe access using .get() for a key that might not exist
print(f"Student's GPA: {student.get('gpa', 'Not Available')}")

# Add a new key-value pair
student['grad_year'] = 2024

# Update an existing value
student['age'] = 22
print(f"Updated dictionary: {student}")

# Iterate over key-value pairs using .items()
print("\\nStudent Details:")
for key, value in student.items():
    print(f"- {key.title()}: {value}")`,
            output: `Student's GPA: Not Available
Updated dictionary: {'name': 'Alice', 'age': 22, 'major': 'Data Science', 'grad_year': 2024}

Student Details:
- Name: Alice
- Age: 22
- Major: Data Science
- Grad_Year: 2024`
        },
        {
            title: "1.3 Operations on Data Structures: A Comparison",
            content: `Understanding which operations are efficient for each data structure is key to writing performant Python code for data science.

**List**
- **Access by Index (\`my_list[i]\`):** Very Fast (O(1)).
- **Membership Test (\`x in my_list\`):** Slow (O(n)). Has to check every element.
- **Add/Remove Items:** Fast at the end (\`.append()\`, \`.pop()\`), but Slow at the beginning or middle (\`.insert()\`, \`.pop(0)\`) because all other elements must be shifted.
- **Best For:** Ordered sequences of items that you might need to change. When you need to access elements by their position.

**Tuple**
- **Access by Index (\`my_tuple[i]\`):** Very Fast (O(1)).
- **Membership Test (\`x in my_tuple\`):** Slow (O(n)).
- **Add/Remove Items:** Not possible (immutable).
- **Best For:** Protecting data that should not be changed. Returning multiple values from a function. Using as a dictionary key.

**Set**
- **Access by Index:** Not possible (unordered).
- **Membership Test (\`x in my_set\`):** Very Fast (O(1) on average). This is its main strength.
- **Add/Remove Items:** Very Fast (O(1) on average).
- **Best For:** Storing unique items. Extremely fast membership testing. Mathematical set operations (union, intersection).

**Dictionary**
- **Access by Key (\`my_dict[key]\`):** Very Fast (O(1) on average).
- **Membership Test (\`key in my_dict\`):** Very Fast (O(1) on average). Checks for keys, not values.
- **Add/Remove Items:** Very Fast (O(1) on average).
- **Best For:** Storing key-value pairs. Fast lookups when you have a unique identifier (the key). A flexible way to structure data (like JSON).`
        }
    ]
};