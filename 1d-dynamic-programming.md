# Dynamic Programming

## Maximum Product Subarray

https://leetcode.com/problems/maximum-product-subarray/description/

Here we keep in track of the curr_min and curr_max - this is because we have negative digits as well.
If max is multiplied by - positive it gives the max positive - negative it gives the max_negative

```
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        result = max(nums)
        curr_min = curr_max = 1

        for num in nums:
            temp = curr_max
            curr_max = max(num * curr_max, num * curr_min, num)
            curr_min = min(num * temp, num * curr_min, num)
            result = max(result, curr_max)
        
        return result
```

## Climbing Stairs

https://leetcode.com/problems/climbing-stairs/description/

This would be solved using DP:
- Choice of taking either 1 or 2 step - step i can be reached from either step (i-1) or step (i-2)
- Optimization problem where we need to find out the total number of ways

**Only Recursion**
```
class Solution:
    def climbStairs(self, n: int) -> int:
        def solve(n):
            if n <= 1:
                return 1
            return solve(n-1) + solve(n-2)
        return solve(n)
```

Here there would be overlapping sub-problems, for example solve(3) or solve(4) would be needed multiple times and we need not calculate it each time, so we use the technique of memoization.

**Recursion with memoization**
```
class Solution:
    def climbStairs(self, n: int) -> int:
        mem = [-1] * (n+1)
        def solve(n):
            if n <= 1:
                return 1
            if mem[n] == -1:
                mem[n] = solve(n-1) + solve(n-2)
            return mem[n]
        return solve(n)
```
