# Recursion
- Based on choices and decision which modify the input.
- We choose recursion when we notice that there is a decision space. 
- If you can draw a recursive tree for the problem, coding is the easy part.

## Recursion is everywhere
- Tree and Graph is based on recursion
- Recursion has multiple implementation:
  - Dynamic Programming
  - Backtracking
  - Divide and Conquer

## Approaches to solve recursive problems
1) Base Condition - Induction - Hypothesis (Making input smaller)
2) Recursive Tree - Input-Output Method (Taking Decisions)
3) Choice Diagram

### 1. Base Condition - Induction - Hypothesis (IBH) - For problems where decisions is not intuitive - reduce the input size
- Useful in Tree and Linked List problems
- Steps:
  - Design the hypothesis - that the function would work for a reduced/smaller input
  - Perform the induction step - perform the necessary step for the current input - through which hypothesis would become true
  - Base condition is the smallest valid input or largest invalid input

**Print from 1 to n**
- Hypothesis - the function works for (n-1)
- Induction - print n to the output
- Base Condition - if the value reaches 0 - return as this is the largest invalid input

```python
def solve(n):
    if n == 0: # Base condition
        return
    solve(n-1) # Hypothesis - would print from 1..n-1
    print(n) # Induction - print n
```
- Could we use recursive tree in this? the tree would be unbalanced

### 2. Recursive Tree - Input-Output Method:
- Start the tree by passing the input and output
- After taking a decision modify the input and pass to the next level with the output of the decision
- In a recursive tree: **number of branches = number of choices**

**Find all possible substrings**
```mermaid
graph TD
I1[abc,``] --ignore a--> I21[bc,``]
I1 --consider a--> I22[bc, a]
I21 --ignore b--> I31[c,``]
I21 --consider b--> I32[c,b]
I22 --ignore b--> I33[c,a]
I22 --consider b--> I34[c,ab]
I31 --ignore c--> I41[``,``]
I31 --consider c--> I42[``, c]
I32 --ignore c--> I43[``,b]
I32 --consider c--> I44[``,bc]
I33 --ignore c--> I45[``,``]
I33 --consider c--> I46[``,a]
I34 --ignore c--> I47[``,ac]
I34 --consider c--> I48[``,abc]
```
Here we get all the possible substrings as the leaf-nodes of the recursive tree.

## Problems
### Induction-Base-Hypothesis
#### Height of a binary tree
- Hypothesis: On calling the function for any of the child node it would return the height of the child sub-tree
- Induction: The height would be - 1(for the current node) + max(left subtree, right subtree)
- Base Condition: When the node is None

```python
def height(root):
    if root is None: # Base condition
        return 0
    left_subtree_height = height(root.left) # Hypothesis
    right_subtree_height = height(root.right)
    return 1 + max(left_subtree_height, right_subtree_height)
```
#### Sort a stack
- Hypothesis: Calling the sort function after removing the top element - would return it sorted
- Induction: Insert the current element in the correct position in the stack and push the elements greater after it
- Base Condition: The stack has a single element
```python
def sort(stack):
    # Base Condition
    if len(stack) <= 1:
        return
    
    # Reduce Input
    top = stack.pop()

    # Hypothesis
    sort(stack)

    # Induction
    temp_stack = []
    while stack[-1] > top:
        temp_stack.append(stack.pop())
    stack.append(top)
    while temp_stack:
        stack.append(temp_stack.pop())
```
#### Delete middle element of stack
-  Hypothesis:
-  Induction:
-  Base Condition:
```python
``` 
#### Reverse stack
-  Hypothesis:
-  Induction:
-  Base Condition:
```python
``` 
#### [Kth Symbol in Grammar](https://leetcode.com/problems/k-th-symbol-in-grammar)
-  Hypothesis: For input (n+1) we hypothise that we would get the answer, k remains constant
-  Induction: Generate the string for n and pass as the one of the parameter for hypothesis call
-  Base Condition: When we reach `nth` row, return `kth` element 
```python
def solve(n, k):
    def helper(row, k, curr):
        # Base Condition
        if row == n:
            return curr[k]
        # Induction
        next = []
        for char in curr:
            if char == "0":
                next.append("01")
            else:
                next.append("10")
        next = "".join(next)
        # Hypothesis
        return helper(row+1, k, next) 
    return helper(1, k, "0")
``` 
**Another approach:**
-  Hypothesis: For smaller input for row `n-1` and any k value we would get the result.
-  Induction: On observation we could state:
   -  the first half of elements in row `n` is equal to the elements in row `n-1` - if k < mid - call with k
   -  the second half of elements in row `n` is equal to the complement of elements in row `n-1` - if k >= mid call with k-mid and complement the answer 
-  Base Condition: When row == 1 and k == 1 return 0
```python
def solve(n, k):
    if n == 1 and k == 1:
        return 0
    length = 2 ** n
    mid = length // 2
    if k <= mid:
        return solve(n-1, k)
    else:
        return not solve(n-1, k-mid)
``` 
#### Tower of Hanoi
-  Hypothesis: Move `n-1` plates from `s` to `h`
-  Induction: Move nth plate from `s` to `d` and `n-1` plate from `h` to `d`
-  Base Condition: When n reaches 1 move from `s` to `d`
-  Remember that the source, helper and the destination are being used interchangeably. Before each call you need to decide what would be assigned what role.
   -  For induction - source `s`, helper `d`, destination `h` - such that we can move the biggest plate to the destination
   -  For the hypothesis
      -  Moving the last plate from s - source `s`, helper `h`, destination `d` - same as the problem
      -  Moving the `n-1` plates to destination - source `h`, helper `s`, destination `d`
```python
source = 's'
helper = 'h'
destination = 'd'
def solve(source, helper, destination, n):
    # Base condition
    if n==1:
        print(f"Plate moved from {source} to {destination}")
        return
    # Hypothesis
    solve(source, destination, helper, n-1)
    # Induction
    print(f"Plate moved from {source} to {destination}")
    solve(helper, destination, source, n-1)
``` 
### Recursive Tree - Input/Output Method
#### Print all subsets
- **Decision space**: For each element either we include in the result space or not.
- Base condition: When all the elements of the input are already consider and index goes out of bounds. We add the current subset as one of the possible results.

Recursive tree
```mermaid
graph TD  
    A["[1, 2, 3], []"] -->|ignore 1| B["[2, 3], []"]  
    A -->|include 1| C["[2, 3], [1]"]  
      
    B -->|ignore 2| D["[3], []"]  
    B -->|include 2| E["[3], [2]"]  
      
    C -->|ignore 2| F["[3], [1]"]  
    C -->|include 2| G["[3], [1, 2]"]  
      
    D -->|ignore 3| H["[], []"]  
    D -->|include 3| I["[], [3]"]  
      
    E -->|ignore 3| J["[], [2]"]  
    E -->|include 3| K["[], [2, 3]"]  
      
    F -->|ignore 3| L["[], [1]"]  
    F -->|include 3| M["[], [1, 3]"]  
      
    G -->|ignore 3| N["[], [1, 2]"]  
    G -->|include 3| O["[], [1, 2, 3]"]  
  
    H["[], []"]  
    I["[], [3]"]  
    J["[], [2]"]  
    K["[], [2, 3]"]  
    L["[], [1]"]  
    M["[], [1, 3]"]  
    N["[], [1, 2]"]  
    O["[], [1, 2, 3]"]
```
Code
```python
def subsets(nums):
    result = []
    n = len(nums)
    def solve(index, curr):
        if index == n:
            result.append(curr[:])
            return
        # Ignore
        solve(index+1, curr)
        # Include
        curr.append(nums[index])
        solve(index+1, curr)
        curr.pop()
    solve(0, [])
    return result
```
#### Subset Duplicates + Other variants
If there are duplicate elements in the subset, the recursive tree would look like:
```mermaid
graph TD  
    A["1, 1, 2, []"] -->|ignore 1| B["1, 2, []"]  
    A -->|include 1| C["1, 2, [1]"]  
      
    B -->|ignore 1| D["2, []"]  
    B -->|include 1| E["2, [1]"]  
      
    C -->|ignore 1| F["2, [1]"]  
    C -->|include 1| G["2, [1, 1]"]  
      
    D -->|ignore 2| H["[], []"]  
    D -->|include 2| I["[], [2]"]  
      
    E -->|ignore 2| J["[], [1]"]  
    E -->|include 2| K["[], [1, 2]"]  
      
    F -->|ignore 2| L["[], [1]"]  
    F -->|include 2| M["[], [1, 2]"]  
      
    G -->|ignore 2| N["[], [1, 1]"]  
    G -->|include 2| O["[], [1, 1, 2]"]  
  
    H["[], []"]  
    I["[], [2]"]  
    J["[], [1]"]  
    K["[], [1, 2]"]  
    L["[], [1]"]  
    M["[], [1, 2]"]  
    N["[], [1, 1]"]  
    O["[], [1, 1, 2]"]
```
As you observe in the leaf nodes there are duplicate subsets in the answer.
Solution:
- Store the subsets as a set and thus the duplicates would be automatically removed. But still computation is wasted as all the subtrees are computed.
- The issue is not with considering the element, just that when we are ignoring an element, ensure that we escape all the occurences of that particular element.

```mermaid
graph TD  
    A["[1, 1, 2], []"] -->|ignore 1| B["[2], []"]  
    A -->|include 1| C["[1, 2], [1]"]  
      
    B -->|ignore 2| D["[], []"]  
    B -->|include 2| E["[], [2]"]  
      
    C -->|ignore 1| F["[2], [1]"]  
    C -->|include 1| G["[2], [1, 1]"]  
      
    F -->|ignore 2| H["[], [1]"]  
    F -->|include 2| I["[], [1, 2]"]  
      
    G -->|ignore 2| J["[], [1, 1]"]  
    G -->|include 2| K["[], [1, 1, 2]"]  
  
    D["[], []"]  
    E["[], [2]"]  
    H["[], [1]"]  
    I["[], [1, 2]"]  
    J["[], [1, 1]"]  
    K["[], [1, 1, 2]"]
```

Code
```python
def distinct_subsets(nums):
    nums.sort()
    result = []
    n = len(nums)
    def solve(index, curr):
        if index == n:
            result.append(curr[:])
            return
        # consider
        curr.append(nums[index])
        solve(index+1, curr)
        curr.pop()

        # ignore
        while index+1 < n and nums[index] == nums[index+1]:
            index = index + 1
        solve(index+1, curr)
    solve(0, [])
    return result
```

##### Variants:
- Print powerset - same as get all subsets
- Print subsequence- Almost same as subset with order of input being maintained.
  - There are 3 concepts:
    - Substring: the subpart must be contiguous in nature
    - Subset: Choose elements in any order
    - Subsequence: The elements combination added to the result set may not be contiguous in nature
  - So the sort logic that we applied for subsets won't work, but the rest of the logic would remain the same.

#### Permutation with spaces
Choice: To add a space after the current element or not
Base Condition: When all the elements of the input is considered and the index goes out of bound.
```mermaid
graph TD  
    A["abc, 'a'"] -->|add space| B["bc, 'a '"]  
    A -->|no space| C["bc, 'a'"]  
      
    B -->|add space| D["c, 'a b '"]  
    B -->|no space| E["c, 'a b'"]  
  
    C -->|add space| F["c, 'ab '"]  
    C -->|no space| G["c, 'ab'"]  
      
    D -->|no space| H["', 'a b c'"]  
  
    E -->|no space| I["', 'a bc'"]  
  
    F -->|no space| J["', 'ab c'"]  
  
    G -->|no space| K["', 'abc'"]  
  
    H["a b c"]  
    I["a bc"]  
    J["ab c"]  
    K["abc"]  

```
Code:
```python
def permutations_space(s):
    n = len(s)
    result = []
    def solve(index, curr):
        if index == n-1:
            curr.append(s[index])
            result.append("".join(curr))
            curr.pop()
            return
        curr.append(s[index])

        # no space
        solve(index + 1, curr)
        # add space
        curr.append(" ")
        solve(index + 1, curr)
        curr.pop()

        curr.pop()
    solve(index + 1, [])
    return result
```
#### Permutation with case change
Choice: In the given input for each character we have 2 choices: uppercase or lowercase.
Base condition: When the index is out of bound or the end of the input string is reached.
```mermaid
graph TD  
    A["ab, ''"] -->|lowercase a| B["b, 'a'"]  
    A -->|uppercase A| C["b, 'A'"]  
      
    B -->|lowercase b| D["', 'ab'"]  
    B -->|uppercase B| E["', 'aB'"]  
  
    C -->|lowercase b| F["', 'Ab'"]  
    C -->|uppercase B| G["', 'AB'"]  
  
    D["ab"]  
    E["aB"]  
    F["Ab"]  
    G["AB"]
```
Code: Here we would use another approach of handling strings permutations
```python
def permutation_case_change(s):
    n = len(s)
    result = []
    def solve(index, curr):
        if index == n :
            result.append(curr)
            return
        solve(index + 1, curr + s[index].lower())
        solve(index + 1, curr + s[index].upper())
    solve(0, "")
    return result
```
#### Generate all balanced parenthesis
- Base Condition: When both open and close reach 0, the current result is added to the response.
- Choices:
    - If the number of open brackets is greater than zero, we can add an open bracket (.
    - If the number of open brackets is less than the number of close brackets, we can add a close bracket ).

Recursive Tree:
- **n=2**
```mermaid
graph TD  
    A["open: 2, close: 2, curr: ''"] -->|open| B["open: 1, close: 2, curr: '('"]  
      
    B -->|open| C["open: 0, close: 2, curr: '(('"]  
    B -->|close| F["open: 1, close: 1, curr: '()'"]  
  
    C -->|close| D["open: 0, close: 1, curr: '(()'"]  
  
    D -->|close| H["open: 0, close: 0, curr: '(())'"]  
      
    F -->|open| J["open: 0, close: 1, curr: '()('"]  
  
    J -->|close| L["open: 0, close: 0, curr: '()()'"]  
  
    H["(())"]  
    L["()()"]
```

- **n=3**
```mermaid
graph TD  
    A["open: 3, close: 3, curr: ''"] -->|open| B["open: 2, close: 3, curr: '('"]  
      
    B -->|open| C["open: 1, close: 3, curr: '(('"]  
    B -->|close| F["open: 2, close: 2, curr: '()'"]  
  
    C -->|open| D["open: 0, close: 3, curr: '((('"]  
    C -->|close| G["open: 1, close: 2, curr: '(()'"]  
  
    D -->|close| E["open: 0, close: 2, curr: '((()'"]  
  
    E -->|close| I["open: 0, close: 1, curr: '((())'"]  
    I -->|close| K["open: 0, close: 0, curr: '((()))'"]  
  
    G -->|open| J["open: 0, close: 2, curr: '(()('"]  
    G -->|close| L["open: 1, close: 1, curr: '(())'"]  
  
    J -->|close| M["open: 0, close: 1, curr: '(()()'"]  
    M -->|close| N["open: 0, close: 0, curr: '(()())'"]  
  
    L -->|open| O["open: 0, close: 1, curr: '(())('"]  
    O -->|close| P["open: 0, close: 0, curr: '(())()'"]  
  
    F -->|open| Q["open: 1, close: 2, curr: '()('"]  
  
    Q -->|open| R["open: 0, close: 2, curr: '()(('"]  
    Q -->|close| S["open: 1, close: 1, curr: '()()'"]  
  
    R -->|close| T["open: 0, close: 1, curr: '()(()'"]  
    T -->|close| U["open: 0, close: 0, curr: '()(())'"]  
  
    S -->|open| V["open: 0, close: 1, curr: '()()('"]  
    V -->|close| W["open: 0, close: 0, curr: '()()()'"]  
  
    K["((()))"]  
    N["(()())"]  
    P["(())()"]  
    U["()(())"]  
    W["()()()"]  
```
Code
```python
def generateParenthesis(n):
    result = []
    def solve(open, close, curr):
        if open == 0 and close == 0:
            result.append(curr)
            return

        if open > 0:
            solve(open-1, close, curr+"(")
        
        if open < close:
            solve(open, close-1, curr+")")

    solve(n, n, "") 
    return result   
```
#### Josephus problem
