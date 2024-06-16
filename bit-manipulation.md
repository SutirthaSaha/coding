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
