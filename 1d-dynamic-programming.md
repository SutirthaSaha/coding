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