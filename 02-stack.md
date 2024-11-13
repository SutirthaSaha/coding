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

### [Maximum Area Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram)*
Given an array representing the heights of bars in a histogram, find the area of the largest rectangle that can be formed within the bounds of the histogram.

Example:
Input: [2, 1, 5, 6, 2, 3]  
Output: 10

#### Hint
This would an application of both `NSL` and `NSR`, and thus including the current histogram bar we can calculate the maximum area.

Code
```python
def largestRectangleArea(heights):
    n = len(heights)
    nsl, nsr = [-1] * n, [n] * n

    # Next smaller to the left - identify the maximum left you can go for your current height
    stack = []
    for index in range(n):
        while stack and heights[stack[-1]] >= heights[index]:
            stack.pop()
        if stack:
            nsl[index] = stack[-1]
        stack.append(index)
    
    # Next larger to the right - identify the maximum right you can go for your current height
    stack = []
    for index in range(n-1, -1, -1):
        while stack and heights[stack[-1]] >= heights[index]:
            stack.pop()
        if stack:
            nsr[index] = stack[-1]
        stack.append(index)
    
    result = 0
    for index, height in enumerate(heights):
        bars = nsr[index] - nsl[index] - 1
        result = max(result, height * bars)

    return result
```

### [Rain Water Trapping](https://leetcode.com/problems/trapping-rain-water)*
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

### [Implementing a Min Stack](https://leetcode.com/problems/min-stack)*
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
##### Intuition
For the implementation without using an auxiliary stack, we can use a clever trick to store the minimum value directly within the main stack.

The key intuition is to store a `transformed` value in the stack when a new minimum value is encountered. This transformed value encodes both the new minimum and the previous minimum. By doing this, we can retrieve the previous minimum value when the new minimum is popped off the stack.

- **Push Operation**:
  - If the stack is empty, the first pushed value is also the minimum value.
  - When pushing a new value, we compare it with the current minimum value (`min_val`):
    - If the new value is greater than or equal to the current minimum, we push it directly onto the stack.
    - If the new value is less than the current minimum, we need to update the minimum. To do this, we push a "marker" value onto the stack. This marker value is calculated as 2 * val - `min_val`. This transformation helps us encode both the new minimum and the old minimum.
  - We then update `min_val` to the new value.

- **Pop Operation**:
  - When popping a value, we check if the popped value is less than the current `min_val`. This indicates that the popped value is a marker for the minimum.
  - To retrieve the previous minimum, we use the formula previous_min = 2 * current_min - marker_value. This formula reverses the transformation we applied during the push operation.
  - We then update `min_val` to this previous minimum.

- **Top Operation**:
  - When retrieving the top value, if the top value is less than the current `min_val`, it means it is a marker for the minimum. Thus, we return `min_val` instead of the top value.
  - Otherwise, we return the top value directly.

Code
```python
class MinStack:  
    def __init__(self):  
        self.stack = []  
        self.min_val = None  
  
    def push(self, val):  
        if not self.stack:  
            self.stack.append(val)  
            self.min_val = val  
        else:  
            if val < self.min_val:  
                # Push a "marker" value that encodes both the new minimum and the old minimum  
                self.stack.append(2 * val - self.min_val)  
                self.min_val = val  
            else:  
                self.stack.append(val)  
  
    def pop(self):  
        if not self.stack:  
            return  
          
        top = self.stack.pop()  
        if top < self.min_val:  
            # This means the current top is a marker for the minimum value  
            self.min_val = 2 * self.min_val - top  
  
    def top(self):  
        if not self.stack:  
            return None  
          
        top = self.stack[-1]  
        if top < self.min_val:  
            return self.min_val  
        else:  
            return top  
  
    def getMin(self):  
        return self.min_val  
```

### [Daily Temperatures](https://leetcode.com/problems/daily-temperatures)*
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

### [Car Fleet](https://leetcode.com/problems/car-fleet)*
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

#### Intuition
To implement a stack using a heap (priority queue), we need to simulate the stack's Last-In-First-Out (LIFO) behavior using the heap's properties. We can use a tuple (priority, element) where priority is a timestamp or counter to ensure the stack order.
We would be using a max-heap and the counter would ensure that the one that is added last is at the top of the heap.

Code
```python
class Stack:
    def __init__(self):
        self.heap = []
        self.counter = 0
    def push(self, val):
        self.counter = self.counter + 1
        heapq.heappush(self.heap, (-self.counter, val))
    
    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        return None
    
    def top(self):
        if self.heap:
            return self.heap[0][1]
        return None
```

### [Longest Valid Parenthesis](https://leetcode.com/problems/longest-valid-parentheses)*
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

#### Intuition
- Use a stack to keep track of the indices of the characters.
- Push the index of the last unmatched `)` onto the stack. Initialize the stack with -1 to handle the edge case for the first valid substring.
- As you iterate through the string, push the index of '(' onto the stack.
- When you encounter ')', pop the stack:
  - If the stack is empty after popping, push the current index onto the stack as the new base for future valid substrings.
  - If the stack is not empty, calculate the length of the current valid substring using the difference between the current index and the index now at the top of the stack.

Code
```python
def longestValidParentheses(s):  
    stack = [-1]  # Initialize stack with -1 to handle edge cases  
    max_length = 0  
  
    for i in range(len(s)):  
        if s[i] == '(':  
            stack.append(i)  # Push the index of '(' onto the stack  
        else:  
            stack.pop()  # Pop the stack for ')'  
            if not stack:  
                stack.append(i)  # Push the current index as the new base - would reach here only if there is an extra )fo  
            else:  
                max_length = max(max_length, i - stack[-1])  # Calculate the length of the current valid substring  
  
    return max_length
```

### Iterative Tower of Hanoi
Solve the Tower of Hanoi problem iteratively. Given three rods (source, auxiliary, and destination) and n disks, where each disk has a different size, move all the disks from the source rod to the destination rod following these rules:
- Only one disk can be moved at a time.
- A disk can only be moved to the top of another rod if it is smaller than the top disk on that rod.
- A disk can only be moved if it is the top disk on a rod.

Implement a function iterativeHanoi(n, source, auxiliary, destination) that prints the steps to move the disks.

#### Intuition
The Tower of Hanoi problem can be solved iteratively using a non-recursive approach. The iterative solution leverages the fact that the pattern of moves is periodic and can be generated using a systematic approach. Here's how you can do it:

- Total Moves: The total number of moves required to solve the Tower of Hanoi problem with n disks is `2^n - 1`.
- Move Pattern: 
  - For an even number of disks, the moves follow the pattern: source -> auxiliary, source -> destination, auxiliary -> destination. 
  - For an odd number of disks, the moves follow the pattern: source -> destination, source -> auxiliary, destination -> auxiliary.
- Using Stacks: 
  - Use three stacks to represent the rods and manage the disks.
  - Keep track of the moves and systematically move the top disks between the rods according to the pattern.

Code
```python
def iterativeHanoi(n, source, auxiliary, destination):  
    # Initialize the rods as stacks  
    rods = {  
        source: list(range(n, 0, -1)),  # Source rod with disks n to 1  
        auxiliary: [],  
        destination: []  
    }  
  
    # Function to print the move  
    def print_move(from_rod, to_rod):  
        print(f"Move disk from {from_rod} to {to_rod}")  
  
    # Function to move the top disk from one rod to another  
    def move_disk(from_rod, to_rod):  
        disk = rods[from_rod].pop()  
        rods[to_rod].append(disk)  
        print_move(from_rod, to_rod)  
  
    # Determine the sequence of moves based on the number of disks  
    if n % 2 == 0:  
        moves = [(source, auxiliary), (source, destination), (auxiliary, destination)]  
    else:  
        moves = [(source, destination), (source, auxiliary), (destination, auxiliary)]  
  
    total_moves = 2 ** n - 1  
  
    # Perform the moves iteratively  
    for i in range(1, total_moves + 1):  
        from_rod, to_rod = moves[(i - 1) % 3]  
          
        # Determine which move to make  
        if rods[from_rod] and (not rods[to_rod] or rods[from_rod][-1] < rods[to_rod][-1]):  
            move_disk(from_rod, to_rod)  
        else:  
            move_disk(to_rod, from_rod)
```

### Generic Stack Problems:
#### [Valid Parenthesis](https://leetcode.com/problems/valid-parentheses)*
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

#### [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation)*
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
