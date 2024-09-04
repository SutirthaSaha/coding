# Backtracking
Is based on `recursion`, where you call the same function again with a reduced or modified set of inputs.

## Basics
- If dynamic programming is `recursion + memoization` then backtracking is `recursion + control` - here control means certain condition must satisfy for the next recursive call input.
- Backtracking questions are usually when you are asked to find **all possible combinations** - the answer lies in the different paths that have satisfied the conditions.
- Along with control, it also involves `pass by reference` where the value modified remains as the same object is modified - so we must ensure that any changes are always `reverted`.

## Remember
Backtracking is the combination of the following:
- Controlled recursion - choices must satisfy a **condition**
- Pass By Reference - same object is being passed and ensure **reversion** of any changes

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
Psuedocode
```
def solve(variables...) -> void:
    # base condition
    if is_solved():
        print or save the copy in global variable
        return
    for choice in choices:
        if is_valid(choice):
            solve()
```

## Problems

### Permutations of String
Take each index and swap with all the postions
Ensure to revert back the changes in order to cover all possible scenarions

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

### Largest number in K swaps


### Rat in a maze

### Word Break

### Letter Combination of Phone Number

### N Queens

### Soduko Solver

