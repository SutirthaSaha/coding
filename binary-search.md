# Binary Search

## Find minimum in Rotated Sorted Array

https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/

Since the array is already sorted - so we can use the binary search property to find the max in `O(logn)` instead of O(n).
However it wouldn't be straight forward as the array is also rotated.
So the idea would be find if the array is already sorted from a pivot and the leftmost would be the minimum.

Complexity: space- O(1) time- O(logn)

```
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n-1

        # the exit condition would be that the array is always rotated
        while nums[l] > nums[r]:
            mid = (r-l) // 2 + l
            if nums[l] <= nums[mid]:
                l = mid + 1
            else:
                r = mid
        
        return nums[l]
```

## Search in a Rotated Sorted Array

https://leetcode.com/problems/search-in-rotated-sorted-array/

```
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        l, r = 0, n-1

        while l<=r:
            mid = (r-l)//2 + l
            if nums[mid] == target:
                return mid
            if nums[l] <= nums[mid]:
                if nums[l] > target or target > nums[mid]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                if nums[r] < target or target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
        
        return -1
```