# Algorithms you must know
## Knuth Morris Pratt (KMP) algorithm

The Knuth-Morris-Pratt (KMP) algorithm is an efficient string matching algorithm that improves upon the naive approach by avoiding unnecessary comparisons. It preprocesses the pattern to create a partial match table (also known as the "prefix" table) that allows the algorithm to skip sections of the text, leading to faster search times.

Example:
```  
text = "abxabcabcaby"  
pattern = "abcaby"  
Output: 6  
```

### Naive Approach
The naive approach to string matching involves checking each position in the text to see if the pattern matches. This approach has a time complexity of O(n * m), where n is the length of the text and m is the length of the pattern.

```python
def naive_search(text, pattern):  
    n = len(text)  
    m = len(pattern)  
      
    for i in range(n - m + 1):  
        match = True  
        for j in range(m):  
            if text[i + j] != pattern[j]:  
                match = False  
                break  
        if match:  
            return i  
    return -1 
```

### Optimal Approach
The KMP algorithm improves upon the naive approach by preprocessing the pattern to create a partial match table (prefix table), which allows the algorithm to skip sections of the text. This preprocessing step has a time complexity of O(m), and the search step has a time complexity of O(n), making the overall time complexity O(n + m).

Steps:
- Preprocessing (Build the Prefix Table):
  - Create a prefix table that indicates the longest proper prefix of the pattern that is also a suffix.
  - This table helps determine the next positions to check in the text when a mismatch occurs.
- Searching: Use the prefix table to skip sections of the text when mismatches occur, avoiding unnecessary comparisons.

Code
```python
def build_prefix_table(pattern):  
    m = len(pattern)  
    prefix_table = [0] * m  
    j = 0  # Length of the previous longest prefix suffix  
      
    for i in range(1, m):  
        while j > 0 and pattern[i] != pattern[j]:  
            j = prefix_table[j - 1]  
        if pattern[i] == pattern[j]:  
            j += 1  
        prefix_table[i] = j  
    return prefix_table

def kmp_search(text, pattern):  
    n = len(text)  
    m = len(pattern)  
    prefix_table = build_prefix_table(pattern)  
    j = 0  # Index for pattern  
      
    for i in range(n):  
        while j > 0 and text[i] != pattern[j]:  
            j = prefix_table[j - 1]  
        if text[i] == pattern[j]:  
            j += 1  
        if j == m:  
            return i - m + 1  
    return -1
```
