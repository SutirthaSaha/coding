# Greedy

## Maximum Subarray

https://leetcode.com/problems/maximum-subarray/description/

Find the maximum subarray sum from an array containing both +ve and -ve integers (obvious because if there were only +ve - the entire array would be giving the max sum)

Brute force would be to calculate the sum of all the subarrays and return the maximum one - would be an `O(n^2)` solution.

We can think of the solution as a sliding window where if the curr element is already a better sub-array than the ones before constituting it we can ignore the earlier sub-array.

Why Greedy?
This is an optimization problem where we are aware of an optimized approach of solving the problem.

```
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        curr_sub, max_sub = nums[0], nums[0]

        for num in nums[1:]:
            curr_sub = max(num, curr_sub+num)
            max_sub = max(max_sub, curr_sub)
        
        return max_sub
```

Also it can be approached in another way thinking that we never let our sub-array be negative sum and assign it to 0 if we see it become 0. But since an empty subarray is not an expected response the max subarray is always assigned as the first value.

```
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        curr_sub, max_sub = 0, nums[0]

        for num in nums:
            if curr_sub < 0:
                curr_sub = 0
            curr_sub = curr_sub + num
            max_sub = max(max_sub, curr_sub)

        return max_sub
```



