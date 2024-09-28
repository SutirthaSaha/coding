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
### Nearest Greater to Right (`NGR`) - Next Larger Element
Given an array of integers, find the nearest greater element to the right of each element in the array. If no such element exists, return -1 for that position.

Example:
Input: [4, 5, 2, 10, 8]  
Output: [5, 10, 10, -1, -1]  

#### Brute force:
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
- If there are no element which is greater than the current element in the stack we set the `NGR` as -1.

**Do we need to do this for all elements?**
No, if we start traversing and setting the `NGR` for each element from the right. We would have the element larger than current already in the stack.
For this we have to also ensure that we do insert the current element in the stack after fidning the `NGR`.

**Concept: Monotonic Stack**
In the above example we would always see a stack in which values are decreasing as the current element would be removing all the elements till it encounters an element which is greater. Such a stack in which the values are following a pattern - either increasing or decreasing - are called **monotonic stack**.

In the monotonic decreasing stack the current element would be lesser than the element below it. So for solving next greater element - the stack would be monotonically decreasing as we want the element which is just greater than the current element and otherwise we anyways pop the stack.

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
### Nearest Greater to Left (`NGL`)
Given an array of integers, find the nearest greater element to the left of each element in the array. If no such element exists, return -1 for that position.

Example:
Input: [4, 5, 2, 10, 8]  
Output: [-1, -1, 5, -1, 10] 

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
### Nearest Smaller to Right (`NSR`) - Next Smaller Element
Given an array of integers, find the nearest smaller element to the right of each element in the array. If no such element exists, return -1 for that position.

Example:
Input: [4, 5, 2, 10, 8]  
Output: [2, 2, -1, 8, -1]

Here there would be a twist - instead of removing the elements which are smaller than the current element we would now be popping elements which are greater - to find the next smaller element.
The resultant would again be a **monotonic stack** but this time it would be an increasing one as the current element would pop all the elements which are greater than it which would result in a stack where the current element would always we greater than the element below it, thus **monotonic increasing stack**.

```python
def nsr(nums):
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(n-1, -1, -1):
        # While stack is not empty and the top of the stack is greater than the current element
        while stack and stack[-1] > nums[i]:
            stack.pop()
        
        # If stack is not empty, the top of the stack is the nearest smaller element to the right
        if stack:
            result[i] = stack = [-1]
        
        # Push the current element onto the stack
        stack.append(nums[i])
    
    return result
```

### Nearest Smaller to Left (`NSL`)
Given an array of integers, find the nearest smaller element to the left of each element in the array. If no such element exists, return -1 for that position.

Example:
Input: [4, 5, 2, 10, 8]  
Output: [-1, 4, -1, 2, 2]

Again the only modification from the previous `NSR` problem would be that we now start from the left.

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
Given an array of daily stock prices, calculate the span of stock’s price for all days. The span of a stock’s price on a given day is the maximum number of consecutive days just before the given day, for which the price of the stock on the current day is less than or equal to its price on the given day.

Example:
Input: [100, 80, 60, 70, 60, 75, 85]  
Output: [1, 1, 1, 2, 1, 4, 6]

#### Hint
This is an application of `NGL`, such that you can get the consecutive days between that and today - giving you the stock span.

### Maximum Area Histogram*
Given an array representing the heights of bars in a histogram, find the area of the largest rectangle that can be formed within the bounds of the histogram.

Example:
Input: [2, 1, 5, 6, 2, 3]  
Output: 10

#### Hint
This would an application of both `NSL` and `NSR`, and thus including the current histogram bar we can calculate the maximum area.

### Rain Water Trapping*
Given an array of non-negative integers representing the height of bars in a histogram, find the total amount of water that can be trapped between the bars after raining.

#### Naive
##### Intution
- Iterate through each index and calculate the water trapped at that index as the minimum of the maximum heights to its left and right minus the height at that index.
- Sum up the trapped water at all indices.

```python
def trap(self, height: List[int]) -> int:  
    n = len(height)  
      
    # Arrays to store the maximum heights to the left and right of each index  
    max_left, max_right = [0] * n, [0] * n  
  
    # Fill the max_left array  
    for i in range(1, n):  
        max_left[i] = max(max_left[i - 1], height[i - 1])  
  
    # Fill the max_right array  
    for i in range(n - 2, -1, -1):  
        max_right[i] = max(max_right[i + 1], height[i + 1])  
  
    # Calculate the total trapped water  
    trapped_water = 0  
    for i in range(n):  
        # Water trapped at index i  
        water_at_i = max(min(max_left[i], max_right[i]) - height[i], 0)  
        trapped_water += water_at_i  
  
    return trapped_water
```

#### Two Pointer Approach
Check the Two Pointer Section
#### Monotonic Stack
##### Intuition
- To trap water, you need to find the left and right boundaries for each bar. The stack helps in maintaining these boundaries as we traverse the array.
- We use a **decreasing monotonic stack**, each element in the stack is lesser than or equal to than the element below it. For an element greater than the top of the stack it now acts as `valley` where water can be trapped.
- The height of the water trapped is determined by the shorter of the two boundaries - (new TOS and the current element) minus the height of the bar that was just removed from the stack.

```python
def trap(height):
    n = len(height)
    stack = []
    water_trapped = 0
    current = 0

    for i in range(n):
        # While the stack is not empty and the current height is greater than the height at the top of the stack
        while stack and height[current] > height[stack[-1]]:
            top = stack.pop()

            # Check if a left boundary exists - if not water cannot be trapped
            if not stack:
                break
            
            # Calculate the distance between the current element and the new top of the stack
            distance = current - stack[-1] - 1
            bounded_height = min(height[current], height[stack[-1]]) - height[top]

            # Calculate the trapped water as bounded height times distance and add to the total
            water_trapped = water_trapped + bounded_height * distance
    
        stack.append(current)
    
    return water_trapped
```

### Implementing a Min Stack*
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:
- MinStack() initializes the stack object.
- void push(int val) pushes the element val onto the stack.
- void pop() removes the element on the top of the stack.
- int top() gets the top element of the stack.
- int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function. 

Example:
Input
```
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
```
Output
```
[null,null,null,null,-3,null,0,-2]
```
Explanation
```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
```

Has 2 versions of implementation:
- With Auxiliary Stack (With Extra Space)
- Without Auxiliary Stack (With O(1) Space)

#### With Auxiliary Stack (with Extra Space)
- To implement a MinStack that supports push, pop, top, and getMin operations in constant time, we can use an auxiliary stack to keep track of the minimum elements.
- This auxiliary stack will store the minimum values at each level of the main stack.

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        # Push the value onto the main stack
        self.stack.append(val)

        # Push the minimum value onto the min_stack
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        val = self.stack.pop()
        if self.min_stack[-1] == val:
            self.min_stack.pop()
    
    def top(self):
        return self.stack[-1]
    
    def get_min(self):
        return self.min_stack[-1]
```

#### Without Auxiliary Stack (O(1) space)
```
TODO
```

### Daily Temperatures
Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.

Example 1:
```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```
Example 2:
```
Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]
```
Example 3:
```
Input: temperatures = [30,60,90]
Output: [1,1,0]
```

#### Hint
As you can understand from the problem it would be a `NGR` problem and we would be using a monotonic decreasing stack.

```python
def daily_temperatures(temperatures):
    n = len(temperatures)
    stack = [n-1]
    result = [0] * n

    for i in range(n-2, -1, -1):
        # While stack is not empty and the top of the stack has temperature lesser than or equal to the current day
        while stack and temperatures[stack[-1]] <= temperatures[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1] - i
        stack.append(i)
    
    return result
```

### Car Fleet*
There are n cars at given miles away from the starting mile 0, traveling to reach the mile target. You are given two integer arrays position and speed, both of length n, where position[i] is the starting mile of the i-th car and speed[i] is the speed of the i-th car in miles per hour.

A car cannot pass another car, but it can catch up and then travel next to it at the speed of the slower car. A car fleet is a car or cars driving next to each other. The speed of the car fleet is the minimum speed of any car in the fleet. If a car catches up to a car fleet at the mile target, it will still be considered as part of the car fleet.

Return the number of car fleets that will arrive at the destination.

Example 1:
```
Input: target = 12, position = [10, 8, 0, 5, 3], speed = [2, 4, 1, 1, 3]  
Output: 3  
Explanation:  
- The cars starting at 10 (speed 2) and 8 (speed 4) become a fleet, meeting each other at 12. The fleet forms at the target.  
- The car starting at 0 (speed 1) does not catch up to any other car, so it is a fleet by itself.  
- The cars starting at 5 (speed 1) and 3 (speed 3) become a fleet, meeting each other at 6. The fleet moves at speed 1 until it reaches the target.
```

Example 2:
```
Input: target = 10, position = [3], speed = [3]  
Output: 1  
Explanation:  
There is only one car, hence there is only one fleet.
```

Example 3:
```
Input: target = 100, position = [0, 2, 4], speed = [4, 2, 1]  
Output: 1  
Explanation:  
- The cars starting at 0 (speed 4) and 2 (speed 2) become a fleet, meeting each other at 4. The car starting at 4 (speed 1) travels to 5.  
- Then, the fleet at 4 (speed 2) and the car at position 5 (speed 1) become one fleet, meeting each other at 6. The fleet moves at speed 1 until it reaches the target.
```

#### Intuition
- If a faster car starts behind a slower car, it will eventually catch up to the slower car and travel at the slower car's speed. This creates a fleet. For each car, calculate how long it will take to reach the target. This is given by (target - position[i]) / speed[i]. This calculation forms the basis for fleet formation.
- Sort the cars by their starting positions in descending order, this crucial as cars **cannot overtake** and the further cars even if they are faster with lesser target time would be part of the fleet before them.
- If the current car's time to reach the target is greater than the previous fleet's time(TOS), create a new fleet.
- Thus the length of the stack would be number of fleets.

```python
def car_fleet(target, position, speed):
    n = len(position)
    cars = [(position[i], (target - position[i]) / speed[i]) for i in range(n)]

    # Sort cars by their starting positions in descending order
    cars.sort(reverse=True)
    stack = [cars[0][1]] # To keep track of fleet times

    for _, time in cars[1:]:
        # If the current car's time is greater than the time at the top of the stack 
        if stack[-1] < time:
            stack.append(time)
    
    return len(stack)
```

### Implementing Stack using Heap
Design a stack using a heap data structure. Implement the following operations:
- push(x): Push element x onto the stack.
- pop(): Removes the element on the top of the stack.
- top(): Get the top element of the stack.
- empty(): Return whether the stack is empty.

You should use a heap to implement these operations and ensure that all operations are performed in O(log n) time complexity.

Example:
```
Input:  
["StackUsingHeap", "push", "push", "top", "pop", "top", "empty"]  
[[], [1], [2], [], [], [], []]  
  
Output:  
[null, null, null, 2, null, 1, false]  
  
Explanation:  
StackUsingHeap stack = new StackUsingHeap();  
stack.push(1);  
stack.push(2);  
stack.top();   // Returns 2  
stack.pop();  
stack.top();   // Returns 1  
stack.empty(); // Returns false
```

```
TODO
```

### The Celebrity Problem
In a party of n people, a celebrity is defined as someone who is known by everyone but knows no one. You are given a matrix M of size n x n where M[i][j] is 1 if person i knows person j, otherwise it is 0. Implement a function findCelebrity that determines if there is a celebrity in the party. If there is a celebrity, return their index (0-based index). If there is no celebrity, return -1.

Example:
```
Input:  
M = [[0, 1, 0],  
     [0, 0, 0],  
     [0, 1, 0]]  
Output: 1  
```
Explanation:  
Person 1 is known by everyone but does not know anyone.

```
TODO
```

### Longest Valid Parenthesis
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

```
TODO
```

### Iterative Tower of Hanoi
Solve the Tower of Hanoi problem iteratively. Given three rods (source, auxiliary, and destination) and n disks, where each disk has a different size, move all the disks from the source rod to the destination rod following these rules:
- Only one disk can be moved at a time.
- A disk can only be moved to the top of another rod if it is smaller than the top disk on that rod.
- A disk can only be moved if it is the top disk on a rod.

Implement a function iterativeHanoi(n, source, auxiliary, destination) that prints the steps to move the disks.

```
TODO
```

### Generic Stack Problems:
#### Valid Parenthesis*
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
- Open brackets must be closed by the same type of brackets.
- Open brackets must be closed in the correct order.
- Every close bracket has a corresponding open bracket of the same type.

##### Intuition
We can use the LIFO property of stacks to ensure that every open bracket is properly closed in the correct order.
**Steps:**
- Push open brackets into the stack
- For the close bracket check whether the TOS matches:
  - If yes - pop
  - No - return `False`
- At the end check if the stack is empty - if empty it is valid parenthesis

```python
def is_valid(s):
    stack = []
    bracket_map = {')': '(', '}': '{', ']': '['}

    for char in s:  
        if char in bracket_map:
            # Check if the TOS is equal to the opening bracket of the current closing bracket  
            if stack and stack[-1] == bracket_map[char]:
                stack.pop()
            else:  
                return False  
        else:
            # If it is an open bracket, push it onto the stack  
            stack.append(char)  
    
    # If the stack is empty, all brackets were properly closed; otherwise, they were not  
    return not stack 
``` 

#### Evaluate Reverse Polish Notation*
You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.

Evaluate the expression. Return an integer that represents the value of the expression.

Note that:
- The valid operators are '+', '-', '*', and '/'.
- Each operand may be an integer or another expression.
- The division between two integers always truncates toward zero.
- There will not be any division by zero.
- The input represents a valid arithmetic expression in a reverse polish - notation.
- The answer and all the intermediate calculations can be represented in a 32-bit integer.

##### Intuition:
To evaluate an arithmetic expression given in Reverse Polish Notation (RPN), we can use a stack. RPN, also known as postfix notation, is a mathematical notation where every operator follows all of its operands. 
For example, the expression "3 4 + 2 * 7 /" is equivalent to "((3 + 4) * 2) / 7" in infix notation.

##### Steps:
- **Push Operands**: If the token is an operand (number), push it onto the stack.
- **Evaluate Operators**: If the token is an operator, pop the necessary number of operands from the stack, perform the operation, and push the result back onto the stack.

```python
def evalRPN(tokens):  
    stack = []  
  
    for token in tokens:  
        if token not in "+-*/":  
            # Push the operand onto the stack  
            stack.append(int(token))  
        else:  
            # Pop the last two operands for the operator  
            right = stack.pop()  
            left = stack.pop()  
            if token == '+':  
                result = left + right  
            elif token == '-':  
                result = left - right  
            elif token == '*':  
                result = left * right  
            elif token == '/':  
                # Integer division that truncates toward zero  
                result = int(left / right)  
            # Push the result of the operation onto the stack  
            stack.append(result)  
  
    # The final result should be the only element left in the stack  
    return stack.pop()
```
