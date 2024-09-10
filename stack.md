# Stack
Stack is a data-structure where the elements follow `FIFO(First In  First Out)` for insertion and deletion.
But when to use this?

## Identification
- There is an array as input.
- Brute force solution is `O(n^2)` and the second loop is dependent on the first one.

**For example**
```python
for i in range(n):
    for j in range(i+1, n):
```

In this case the solution can be made linear and improvised using stack.

## Problems
### Nearest Greater to Right (NGR) - Next Larger Element
Brute force:
```python
def ngl(nums):
    n = len(nums)
    result = [-1] * n
    for i in range(n):
        for j in range(i+1, n):
            if nums[j] > nums[i]:
                result[i] = nums[j]
                break
    return result
```
Time Complexity: O(n^2)
And here we can see that the second loop is dependent on the first loop(`j`->`i`).

**How do we use a stack here?**
- We traverse from the right and store all the elements
- Then for the current element we pop the stack till we get an element which is greater than the current element
- If there are no element which is greater than the current element in the stack we set the NGR as -1.

**Do we need to do this for all elements?**
No, if we start traversing and setting the NGR for each element from the right. We would have the element larger than current already in the stack.
For this we have to also ensure that we do insert the current element in the stack after fidning the NGR.

**Concept: Monotonic Stack**
In the above example we would always see a stack in which values are decreasing as the current element would be removing all the elements till it encounters an element which is greater. Such a stack in which the values are following a pattern - either increasing or decreasing - are called **monotonic stack**.

So for solving next greater element - the stack would be monotonically decreasing and vice versa.

Code
```python
def ngr(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    
    # traverse from the right
    for i in range(n-1, -1, -1):
        # pop till finding  a value greater than the current
        while stack and stack[-1] < nums[i]:
            stack.pop()
        
        if stack:
            result[i] = stack[-1]
        
        # push the current value (smaller than the top) to the top of stack
        # the idea is that even though it is smaller than the top, can be larger than the next element
        stack.append(nums[i])
    
    return result
```
### Nearest Greater to Left (NGL)
The only modification from the previous problem would be that we would now need to traverse from left.
The stack would still be monotonic decreasing in nature.

Code
```python
def ngl(nums):
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(n):
        while stack and stack[-1] < n:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(nums[i])
    return result
```
### Nearest Smaller to Right (NSR) - Next Smaller Element
Here there would be a twist - instead of removing the elements which are smaller than the current element we would now be popping elements which are greater - to find the next smaller element.
The resultant would again be a **monotonic stack** but this time it would be an increasing one as the current element would pop all the elements which are greater than it, thus **monotonic increasing stack**.

```python
def nsr(nums):
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(n-1, -1, -1):
        while stack and stack[-1] > nums[i]:
            stack.pop()
        if stack:
            result[i] = stack = [-1]
        stack.append(nums[i])
    
    return result
```

### Nearest Smaller to Left (NSL)
Again the only modification from the previous NSR problem would be that we now start from the left.

```python
def nsl(nums):
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(n):
        while stack and stack[-1] > nums[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(nums[i])
    return result
```

### Stock Span Problem
This is an application of NGL, such that you can get the consecutive days between that and today - giving you the stock span.

### Maximum Area Histogram
This would an application of both NSL and NSR, and thus including the current histogram bar we can calculate the maximum area.

### Rain Water Trapping
This is not a stack problem at all, just keep the prefix maximum(largest element to the left) and the suffix maximum(largest element to the right) and calculate the amount of water that can be stored on the current rooftop. Add for all the buildings would give the total rainwater trapped.

### Implementing a Min Stack
Has 2 versions of implementation:
- With Auxiliary Stack (With Extra Space)
- Without Auxiliary Stack (With O(1) Space)

### Implmenting Stack using Heap
### The Celebrity Problem
### Longest Valid Parenthesis
### Iterative Tower of Hanoi
