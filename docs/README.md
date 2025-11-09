# cuKINSHIP-Lite Documentation

This directory contains generated API documentation for cuKINSHIP-Lite.

## Generating Documentation

### Prerequisites

- [Doxygen](https://www.doxygen.nl/) 1.9.1 or later
- [Graphviz](https://graphviz.org/) (optional, for diagrams)

### Build HTML Documentation

```bash
# From project root
doxygen Doxyfile
```

This will generate HTML documentation in `docs/html/`.

### View Documentation

Open the generated documentation in your browser:

```bash
# Linux/macOS
open docs/html/index.html

# Windows
start docs/html/index.html
```

Or use Python's built-in HTTP server:

```bash
cd docs/html
python3 -m http.server 8000
# Navigate to http://localhost:8000
```

## Documentation Structure

The generated documentation includes:

- **API Reference:** Complete documentation of all public functions, classes, and structures
- **File Index:** Browse source files and their contents
- **Class Hierarchy:** Visual representation of class relationships
- **Call Graphs:** Function call relationships (if Graphviz is installed)
- **Source Browser:** View source code with syntax highlighting

## API Documentation Guidelines

When contributing code, please document:

### Functions

```cpp
/**
 * Brief description of what the function does.
 *
 * More detailed description if needed. Can span
 * multiple lines and include implementation notes.
 *
 * @param param1 Description of first parameter
 * @param param2 Description of second parameter
 * @return Description of return value
 * @throws ExceptionType When this exception is thrown
 *
 * @see related_function
 *
 * Example:
 * @code
 * int result = my_function(10, 20);
 * @endcode
 */
int my_function(int param1, int param2);
```

### Classes

```cpp
/**
 * Brief description of the class.
 *
 * More detailed description of the class purpose,
 * usage patterns, and any important notes.
 *
 * Example:
 * @code
 * MyClass obj;
 * obj.do_something();
 * @endcode
 */
class MyClass {
public:
    /**
     * Brief description of member function.
     */
    void do_something();

private:
    int value_;  ///< Brief description of member variable
};
```

### CUDA Kernels

```cpp
/**
 * Brief description of what the kernel computes.
 *
 * @param input Input data (device pointer)
 * @param output Output data (device pointer)
 * @param n Number of elements
 *
 * @note Thread organization: 1D grid, 1D blocks
 * @note Shared memory usage: X bytes per block
 * @note Assumes: Input and output are properly allocated
 */
__global__ void my_kernel(const float* input, float* output, int n);
```

## Online Documentation

The latest API documentation is available at:
- **Stable:** https://example.com/docs/stable/
- **Development:** https://example.com/docs/dev/

(Update these URLs when documentation hosting is set up)

## Updating Documentation

1. Write code with proper Doxygen comments
2. Run `doxygen Doxyfile` to generate docs
3. Check for warnings and fix them
4. Review generated HTML for correctness
5. Commit source code (not generated docs)

## Troubleshooting

**"Doxygen: command not found"**
```bash
# Ubuntu/Debian
sudo apt-get install doxygen graphviz

# macOS
brew install doxygen graphviz

# Windows
# Download from https://www.doxygen.nl/download.html
```

**Warnings about undocumented functions**
- Add Doxygen comments to public APIs
- Private/internal functions can remain undocumented

**Broken diagrams**
- Install Graphviz: `sudo apt-get install graphviz`
- Ensure `dot` is in your PATH

## Contributing

When adding documentation:
- Use clear, concise language
- Include examples for complex functions
- Document parameters, return values, and exceptions
- Add `@see` tags to link related functions
- Keep line length under 100 characters

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.
