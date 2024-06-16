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

## Product of array except self

https://leetcode.com/problems/product-of-array-except-self/

For each index we have to get the product of its prefixes(elements coming before it) and its suffixes (elements that come after it)

Constraints - to be done in `O(n)` without the division operator

Solution would be to calculate the prefix product for each index and store it in an array and similarly the same for the postfix products for an index.
Then multiply each of these index by index to get the resultant array.

Complexity: space- O(n) time- O(n)

```
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix, postfix = [1] * n, [1] * n

        for i in range(1, n):
            prefix[i] = prefix[i-1] * nums[i-1]
        for i in range(n-2, -1, -1):
            postfix[i] = postfix[i+1] * nums[i+1]
        
        result = []
        for i in range(n):
            result.append(prefix[i]*postfix[i])
        
        return result
```

