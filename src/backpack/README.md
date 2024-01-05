# Backpack Package

## Overview

The **Backpack** local database package serves as a centralized repository for storing and managing diverse data associated with a biased Molecular Dynamics (MD) run. Its primary purpose is to offer a convenient solution for saving and organizing various types of data in one consolidated location. The package provides functionalities for creating, loading, updating, and querying a local database stored as a pickle file. It supports a range of data types, including Pandas DataFrames, NumPy arrays, DotMaps, lists, and other data types, making it a versatile tool for researchers and developers working with MD simulations.

## Usage

### Initialization

To use the `Backpack` package, first, import it into your Python script or module:

```python
from backpack.main import Backpack
```

#### Creating an Instance

```python
# Create a Backpack instance with the default location called backpack.pkl
backpack_instance = Backpack()

# Or specify a custom location for the database file
custom_location = "/path/to/custom/backpack.pkl"
backpack_instance = Backpack(location=custom_location)
```

### Methods

- `set(key, value)` -> Add or update an item in the database.

```python
backpack_instance.set("temperature", 300.0)
```

- `get(key)` -> Retrieve the value associated with a given key.

```python
temperature = backpack_instance.get("temperature")
```

- `delete(key)` -> Remove an item from the database using its key.

```python
backpack_instance.delete("temperature")
```

- `list()` -> Display a summary of the contents of the backpack, including key, data type, content or value, and shape (if applicable).

```python
backpack_instance.list()
```

- `dump()` -> Save the current state of the backpack to the specified file location.

```python
backpack_instance.dump()
```

- `load(location)` -> Load the database from a file location. If no database is found at the specified location, a new one will be created.

```python
backpack_instance.load("/path/to/another/backpack.pkl")
```

### Example

```python
# Create a new Backpack instance
my_backpack = Backpack()

# Add data to the backpack
my_backpack.set("temperature", 300.0)
my_backpack.set("positions", pd.DataFrame(data=np.random.rand(10, 3), columns=['x', 'y', 'z']))

# Display the contents of the backpack
my_backpack.list()
```

## Data Types Supported

- **Pandas DataFrame (`pd.DataFrame`):** If the value is a DataFrame, the column names and shape will be displayed.

- **NumPy Array (`np.ndarray`):** If the value is a NumPy array, the first and last two values, along with the shape, will be shown.

- **List:** If the value is a list, the first and last two elements, along with the shape (converted to a NumPy array), will be displayed.

- **Other Data Types:** For other data types, only the value will be displayed.

## Error Handling

The package provides basic error handling for file operations. If an error occurs during the dumping of the database, an error message will be printed.

```python
# Attempt to dump the database
if not my_backpack.dump():
    print("Error: Could not dump the database.")
```

## Notes

- The package uses the `pickle` module for serialization and deserialization of the database.

- The default location for the database file is the current working directory, but it can be customized during the initialization.

- The `verbose` parameter controls whether informative messages are printed during the execution of certain methods.

---

Feel free to adjust and expand the documentation based on your specific needs and requirements.