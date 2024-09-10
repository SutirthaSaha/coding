# Sliding Window
Problem:
Given an array and a number 'k', return the maxinum sum of all possible subarrays of size 'k'.
Naive Solution:
We can start with each index and loop for 'k' elements and calculate the sum for each index and calculate the maximum.
```python
def find_max_sum_subarray(nums, k):
    n = len(nums)
    max_sum = float('inf')
    for i in range(n - k + 1):
        curr_sum = 0
        for j in range(i, i+3):
            curr_sum = curr_sum + nums[j]
        max_sum = max(max_sum, curr_sum)
    return max_sum
```
This is an `O(n*k)` approach, can we make it better?

## Identification
Check for 3 things:
- input is an array or string
- there is a window size provided or asks for a window size satisfying a particular condition (largest/smallest/unique)
- are we doing any repetitive work?

## Types
There are 2 types of sliding window problem:
- **Fixed Window Size**: The window size is provided as input
- **Dynamic Window Size**: The problem statement requires to find the largest/smallest satisfying a particular condition

Now how do we solve the previous problem. The problem has window size provided as input - fixed sliding window problem.

```python
def find_max_sum_subarray(nums, k):
    n = len(nums)
    max_sum = float('-inf')
    curr_sum = 0
    start = 0
    for end in range(n):
        curr_sum = curr_sum + nums[end]

        if (end - start + 1) > k:
            curr_sum = curr_sum - nums[start]
            start = start + 1
        
        if (end - start + 1) == k:
            max_sum = max(max_sum, curr_sum)
    return max_sum
```
This would be an O(n) solution and we would use the sum calculated in the previous window - thus reducing repetitive calculation.

## Problems
### First -ve number in each window

