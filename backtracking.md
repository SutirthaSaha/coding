# Backtracking
Is based on `recursion`, where you call the same function again with a reduced or modified set of inputs.

## Basics
- If dynamic programming is `recursion + memoization` then backtracking is `recursion + control` - here control means certain condition must satisfy for the next recursive call input.
- Backtracking questions are usually when you are asked to find **all possible combinations** - the answer lies in the different paths that have satisfied the conditions.
- Along with control, it also involves `pass by reference` where the value modified remains as the same object is modified - so we must ensure that any changes are always `reverted`.

## Remember
Backtracking is the combination of the following:
- Controlled recursion - choices must satisfy a **condition**
- Pass By Reference - same object is being passed and ensure **reversion** of any changes or we have a global variable

This reversion is refered as **backtracking**.

## Identification of problems
Search for these keywords in the problem statement (may work)
- choices + decision
- all combinations
- controlled recursion - condition there in choices
- number of choices - the number of choices are dynamic and large compared to recursion
- don't be greedy - (largest number in k swaps problem)

**All backtracking problems can be solved by recursion - but it is more complicated and non-intuitive**

## Generalisation
Psuedocode - Blueprint
```
def solve(variable) -> void:
    # base condition
    if is_solved():
        print or save the copy in global variable
        return
    for choice in choices:
        if is_valid(choice):
            change variable - variable'
            solve(variable')
            revert change in variable - variable
```

## Flow
- IP-OP-PS: Input-Output-Problem Statement
- Choices -> Controlled
  - Out of Bound
  - Repeated
  - Blocked
- BC: Base Condition
- Code

## Problems

### Permutations of String
Flow:
- IP: Given a string
- OP-PS: Return all possible permutations
- Choices:
  - Swap current element with all the next elements
  - No condition in swapping
- Base Condition: Current permutation reaches the end of the string
- Ensure to revert back the changes in order to cover all possible scenarions

Similar problem:
- https://leetcode.com/problems/permutations/description/

Code
```
def permutation(str):
    str_arr = list(str)
    n = len(str_arr)
    result = []
    def solve(index):
        if index == n:
            result.append(str_arr[:])
        for swap_index in range(index, n):
            arr[index], arr[swap_index] = arr[swap_index], arr[index]
            solve(index+1)
            arr[index], arr[swap_index] = arr[swap_index], arr[index]
    solve(0)
    return result
```

If there are duplicates in the string or array there would be a small check that the value to swapped is not the same as the current value.

```
for swap_index in range(index, n):
    if arr[swap_index] != arr[index]: # this would help avoid the redundant sub trees
        arr[index], arr[swap_index] = arr[swap_index], arr[index]
        solve(index+1)
        arr[index], arr[swap_index] = arr[swap_index], arr[index]
```

### Largest number in at most K swaps
Flow:
- IP: Given a number
- OP-PS: Largest number in at most K-swaps
- Choices:
  - Swap with the next digits in the sequence
  - Condition: Swap only if the digit is greater than the current and maximum of the remaining digits
- Base Condition: Either at the end of the number or we complete k swaps
**Points**
- The result is not only present in the base condition at we have to find the maximum   with **at most** k swaps.
- **Horizontal drifting** - after exploring all choices with the current index we drift to the right index without reducing the `k` value.

Code
```
num_arr = list(num_str)
result = float('-inf')
n = len(num_arr)
def solve(index, k):
    result = max(result, int("".join(num_arr)))
    if k == 0 or index == n-1:
        return
    max_value = max(num_arr[index+1:])
    for swap_index in range(index+1, n):
        if num_arr[index] < num_arr[swap_index] and num_arr[swap_index] == max_value:
            num_arr[index], num_arr[swap_index] = num_arr[swap_index], num_arr[index]
            solve(index + 1, k - 1)
            num_arr[index], num_arr[swap_index] = num_arr[swap_index], num_arr[index]
    solve(index+1, k) # Horizontal drifting
```
### N digit number in increasing order
Flow:
- IP- Number N - denotes the number of digits
- OP - Maximum number
- Choices:
  - For 1 digit number we have choices from 0-9
  - For 2+ digits number we have choices from 1-9 - if the first digit is 0 the number automatically is N-1 digit and since the values are always increasing
- Base Condition: N digit number found
- here we pass the current result as `pass by reference` so any changes must be reverted.
```
result = []
def solve(start, curr):
    if N == len(curr):
        result.append(int("".join(curr)))
        return
    if start == 10:
        return
    for digit in range(start, 10):
        curr.append(digit)
        solve(digit+1, curr)
        curr.pop()
if N == 1:
    solve(0, [])
else:
    solve(1, [])
return result
```

### Rat in a maze
Flow:
- IP - Grid and current position(0, 0)
- OP - All possible ways
- PS - From the current position reach the end and grid with value 0 is blocked
- Choices
  - Can move all directions - U(0,-1), D(0, 1), L(-1, 0), R(1, 0)
  - Don't move to invalid positions
    - Matrix position out of bound
    - Matrix value at that position has value 0
    - Can't go to the already traversed point
- Base Condition
  - When the rat reaches target position(m, n) - add the path to the result
- Code
```
result = []
m, n = len(grid), len(grid[0])
direction = [(0, -1), (0, 1), (-1, 0), (1, 0)]
def is_valid(row, col):
    return 0<=m<row and 0<=n<col
def solve(row, col, path):
    if row == m and col == n:
        result.append(path[:])
        return
    for direction in directions:
        n_row, n_col = row + direction[0], col + direction[1]
        if is_valid(n_row, n_col) and grid[n_row][n_col] == 1 and (n_row, n_col) not in visited:
            grid[n_row][n_col] = 0
            path.append((n_row, n_col))
            solve(n_row, n_col, path, visited)
            grid[n_row][n_col] = 1
            path.pop()
solve(0, 0, [], set())
return result
```
### Palindrome Partitioning
Flow:
- IP: String
- OP-PS: All possible palindrome partitions
- Choices:
  - From the start index try possible substrings
  - Ensure that the substrings are palindrome and then only go for the rest of the substrings
- Base Condition: End of the string
- Code
```
def all_palindrome_partitions(string):
    n = len(string)
    result = []

    def is_palindrome(s):
        start, end = 0, len(s) - 1
        while start <= end:
            if s[start] != s[end]:
                return False
        return True 

    def solve(index, curr):
        if index == n:
            result.append(curr[:])
            return
        for end_index in range(index+1, n):
            sub_str = string[index: end_index]
            if is_palindrome(sub_str):
                curr.append(sub_str)
                solve(end_index, curr)
                curr.pop()
    solve(0, [])
    return result
```
### Word Break
Flow:
- IP: A string and a list of words which acts as the dictionary
- OP-PS: Return the words that can be formed from the string in different combination
- Choices:
  - From the start index try all possible substring
  - If any substring exists then only proceed for the rest of the stirng and continue with the same process
- Base Condition: End of the string
- Code
```
def word_break(string, dictionary):
    n = len(string)
    dictionary = set(dictionary)

    result = []
    def solve(index, curr):
        if index == n:
            result.append(curr[:])
            return
        for end_index in range(index+1, n):
            sub_str = string[index: end_index]
            if sub_str in dictionary:
                curr.append(sub_str)
                solve(end_index, curr)
                curr.pop()
    solve(0, [])
    return result
```

### Letter Combination of Phone Number
Flow:
- IP: Digit combination as a string
- OP: All possible letter combination that the number or digit combination can represent
- Choices:
  - For each digit consider all the letter combinations that it can represent
- Base Condition: End of the digits in input
- Code:
```
def letter_combination(digits):
    digit_map = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz" 
    }
    n = len(digits)
    result = []
    def solve(index, curr):
        if index == n:
            result.append("".join(curr))
            return
        digit = digits[index]
        for char in digit_map[digit]:
            curr.append(char)
            solve(index+1, curr)
            curr.pop()
    solve(0, [])
    return result
```

### N Queens
Flow:
- IP: n - denoting n*n board
- OP: all possible board combinations with n queens
- Choices:
  - Place a queen at a position in the board
  - Condition is that the placed queen is safe from any other queen in the board 
- Base Condition: Completed all the rows of the board
- Code
```
def nqueens(n):
    board = [["."] * n for _ in range(n)]
    result = []
    
    def is_safe(row, col):
        # check for column in the previous rows
        for prev_row in range(row):
            if board[prev_row][col] == "Q":
                return False
        # check in the upper left diagonal
        prev_row, prev_col = row-1, col-1
        while prev_row >= 0 and prev_col >=0:
            if board[prev_row][prev_col] == "Q":
                return False
            prev_row = prev_row - 1
            prev_col = prev_col - 1
        # check in the upper right diagonal
        prev_row, prev_col = row-1, col + 1
        while prev_row >= 0 and prev_col < n:
            if board[prev_row][prev_col] == "Q":
                return False
        return True

    def solve(row):
        if row == n:
            result.append(["".join(row) for row in board])
            return
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = "Q"
                solve(row+1)
                board[row][col] = "."
    solve(0)
    return result
```
### Soduko Solver
Similar to N-Queens just the is_safe logic would be for Sudoku

