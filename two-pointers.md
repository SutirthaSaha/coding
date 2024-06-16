# Two Pointers

## Best Time to Buy and Sell Stock

https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

Find the transaction that maximizes the profit - profit can happen when you buy low and sell high - need to find the lowest low that can happen before a high value.
So we can have a pointer at the lowest till a particular day(buy price) and try to maximize the profit with the opther pointer traversing through the days.

Complexity: space- O(1) time- O(n)

```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        buy_price = prices[0]

        for price in prices[1:]:
            profit = price - buy_price
            if profit < 0:
                buy_price = price
            max_profit = max(max_profit, profit)
        
        return max_profit
```

## 3 Sum

https://leetcode.com/problems/3sum/description/

Based on **Two Sum II**: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/

Here sort the input array and then iterate each element - considering the element as a part of the triplet - now the problem becomes a 2Sum problem.
The two pointers here refer to the left and the right pointer which are updated in the following manner:
- Update left when the value in the target is greater
- Update right when the value in the target is lesser

```
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        for index, num in enumerate(nums):
            if index >0 and nums[index-1] == num:
                continue
            l, r = index+1, len(nums)-1
            while l < r:
                triplet_sum = num + nums[l] + nums[r]
                if triplet_sum == 0:
                    result.append([num, nums[l], nums[r]])
                    l = l + 1
                    r = r - 1
                elif triplet_sum > 0:
                    r = r - 1
                else:
                    l = l + 1
        return result
```
## Container With Most Water

https://leetcode.com/problems/container-with-most-water/description/

Here the wall's heights are given and on the basis of the distance between the walls the volume of water contained would vary.
The movement of the pointer is simple in this case whichever wall is shorter amongst the 2 - wouldn't be considered the next time.

```
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        max_area = 0
        while l < r:
            area = min(height[l], height[r]) * (r-l)
            max_area = max(max_area, area)

            if height[l] <= height[r]:
                l = l + 1
            else:
                r = r - 1
        
        return max_area
```
