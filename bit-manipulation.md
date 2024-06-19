# Bit Manipulation

## Number of 1 bits

https://leetcode.com/problems/number-of-1-bits/description/

To find the number of 1 bits the basic approach would be performing a logical AND operation with the number and 1.
If the last bit is 1, then the result would be 1 otherwise 0.
Now for checking the previous bit we can right shift the number by 1.

```
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            if n & 1:
                count = count + 1
            n = n >> 1
        return count
```

## Counting  Bits

https://leetcode.com/problems/counting-bits/description/

This would have a dynamic programming implementation - where the previous results of the power of 2 would be used for calculating the current answer.

```
class Solution:
    def countBits(self, n: int) -> List[int]:
        mem = [0] * (n+1)
        for i in range(1, n+1):
            mem[i] = mem[i>>1] + i%2
        return mem
```

## Missing Number

https://leetcode.com/problems/missing-number/description/

So the brute force way would be to create a hashset with the values from `[0.. n]` and then compare if the value exists in the array and delete it in the set if it doesn't. Takes `O(n)` memory which is execeeding the constraint of `O(1)`.

**Using AP**
Since the values are from 0.. n -> we can also use `Arithmetic Progression` to get the sum for the sequence and then calculate the sum of the array. On finding the difference you would get the missing number.

```
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        sum_nums = sum(nums)
        expected = ((n+1) / 2) * (n)
        return int(expected - sum_nums)
```

**Using XOR**
On performing XOR the duplicate elements cancel out to 0 -> so on calculating the XOR of the array and a sequence from `[0..n]` would give us the missing number as it is the one which is unique.
```
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        res = 0
        for i in range(len(nums)):
            res = res ^ (i+1) ^ nums[i]
        return res
```

## Reverse Bits

https://leetcode.com/problems/reverse-bits/description/

The solution would be to calculate the ith bit from right and set it to (31-i)th bit in the result.

```
class Solution:
    def reverseBits(self, n: int) -> int:
        result = 0

        for i in range(32):
            bit = (n>>i) & 1
            result = result | (bit<<(31-i))
        
        return result
```
