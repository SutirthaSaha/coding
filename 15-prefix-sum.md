# Prefix Sum
Prefix Sum involves preprocessing an array to create a new array where each element at index i represents the sum of the array from the start up to i. This allows for efficient sum queries on subarrays.

Use this pattern when you need to perform multiple sum queries on a subarray or need to calculate cumulative sums.

## Problems
### Range Sum Query
Given an integer array `nums`, handle multiple queries of the following type:

Calculate the sum of the elements of `nums` between indices `left` and `right` inclusive where `left` <= `right`.
Implement the NumArray class:
- `NumArray(int[] nums)`: Initializes the object with the integer array nums.
- `int sumRange(int left, int right)`: Returns the sum of the elements of nums between indices left and right inclusive (i.e. nums[left] + nums[left + 1] + ... + nums[right]).

#### Intuition
- Preprocess the array to create a prefix sum array.
- To find the sum between indices `left` and `right`, use the formula: `P[right] - P[left-1]`.
- If the `left` is `0`, no subtraction is needed.

Code
```
class NumArray:
    def __init__(self, nums: List[int]):
        n = len(nums)
        self.prefix_sum = [nums[0]]
        for num in nums[1:]:
            self.prefix_sum.append(num + self.prefix_sum[-1])

    def sumRange(self, left: int, right: int) -> int:
        if left == 0:  
            return self.prefix_sum[right]  
        else:  
            return self.prefix_sum[right] - self.prefix_sum[left - 1]
```

### Subarray Sum Equals K
Given an array of integers `nums` and an integer `k`, return *the total number of subarrays whose sum equals* to `k`.

A subarray is a contiguous **non-empty** sequence of elements within an array.

Example
```
Input: nums = [1,2,3], k = 3
Output: 2
```

#### Intuition
The intuition behind the solution to this problem is to use a prefix sum (cumulative sum) approach along with a hashmap (dictionary) to keep track of the number of times each prefix sum occurs. 
By maintaining a hashmap of prefix sums, we can efficiently find the number of subarrays that sum up to k by checking if there exists a prefix sum that, when subtracted from the current prefix sum, equals k.

Code
```python    
def subarraySum(self, nums: List[int], k: int) -> int:
    prefix_sum_map = defaultdict(int)
    prefix_sum_map[0] = 1
    n = len(nums)
    count = 0
    prefix_sum = 0
    for i in range(n):
        prefix_sum = prefix_sum + nums[i]
        if (prefix_sum - k) in prefix_sum_map:
            count = count + prefix_sum_map[prefix_sum - k]
        prefix_sum_map[prefix_sum] = prefix_sum_map[prefix_sum] + 1
    return count
```
