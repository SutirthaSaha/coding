# Arrays and Hashing

## Contains Duplicate

https://leetcode.com/problems/contains-duplicate/description/

There can be multiple solutions to the problem of finding duplicates in the array.

**Using sorting** - duplicates will be adjacent
Complexity: space - O(1) time - O(nlogn)
```
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i-1] == nums[i]:
                return True
        return False
```

**Using set** - extra space solution - keep checking in the hashmap if the value already exists
Complexity: space- O(n) time- O(n)
```
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        digits = set()
        for num in nums:
            if num in digits:
                return True
            digits.add(num)
        return False
```

## Two Sum
https://leetcode.com/problems/two-sum/description/

Find out 2 elements in the array such that it sums up to the target.

**Using map** - iterate through the array and store the val(key) and the index(value) - for each value check if there is a previous value which add upto the target and return the result as indices.

Complexity: space- O(n) time- O(n)

```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        prev_map = {}

        for index, value in enumerate(nums):
            diff = target - value
            if diff in prev_map:
                return [prev_map[diff], index]
            prev_map[value] = index
```


