# Greedy
Greedy algorithms are a class of algorithms that build up a solution piece by piece, always choosing the next piece that offers the most immediate benefit. The **hope** is that by making **locally optimal choices** at each step, the **overall solution will be optimal as well**.

## Characteristics
- **Greedy Choice Property**: The algorithm makes a series of decisions by choosing the best available option at each step without considering future consequences. This choice must be locally optimal and irrevocable.
- **Optimal Substructure**: The problem can be broken down into smaller subproblems, and the optimal solution to the problem can be constructed from the optimal solutions to its subproblems.

## Limitations
- **Non-Optimal Solutions**: Greedy algorithms do not always produce the optimal solution. They work well for problems with the greedy choice property, but may fail for others.
- **Local Optima**: Greedy algorithms can get stuck in local optima, missing the global optimal solution.

## Problems
### Activity Selection Problem
Given a set of activities with start and end times, select the maximum number of activities that don't overlap.

#### Strategy
Always select the activity that finishes first. This way, you leave as much room as possible for the remaining activities.

Code
```python
def activity_selection(activities):
    # Sort activities by end time
    activities.sort(key=lambda activity: activity[1])

    selected_activities = [activities[0]]

    for activity in activities[1:]:
        # If the start time of the current activity is greater than or equal to the end time of the last selected activity
        if activity[0] >= selected_activities[-1][1]:
            # Select the current activity
            selected_activities.append(activity)
    
    return selected_activities
```

This problem is similar to the `Non Overlapping Intervals` problem in the intervals section.

### Fractional Knapsack Problem
Given a set of items, each with a weight and a value, determine the maximum value you can carry in a knapsack with a weight limit, where you can take fractions of items.

#### Strategy
Calculate the value-to-weight ratio for each item and sort them in descending order. Then, take as much of the highest ratio item as possible, followed by the next highest, and so on.

Code
```python
def fractional_knapsack(weights, values, capacity):
    n = len(weights)
    
    # Calculate value to weight ratio for each item  
    ratios = [(values[i] / weights[i], weights[i], values[i]) for i in range(n)]

    # Sort items by value to weight ratio in descending order
    ratios.sort(reverse=True, key = lambda ratio: ratio[0])

    total = 0.0
    for ratio, weight, value in ratios:
        if capacity >= weight:
            # Take the whole item
            capacity = capacity - weight
            total_value = total_value + value
        else:
            # Take the fraction of item that fits
            total_value = total_value + ratio * capacity
            break
    
    return total
```

### Job Sequencing Problem
Given a set of jobs where each job has a deadline and a profit, schedule jobs to maximize total profit. Each job takes one unit of time and a job can only be scheduled if it can be completed by its deadline.

#### Strategy
Sort the jobs in descending order of profit. Then, try to schedule each job in the latest possible time slot before its deadline.

Code
```python
def job_sequencing(deadlines, profits, m):
    n = len(deadlines)
    jobs = [(i, deadlines[i], profits[i]) for i in range(n)]

    # Sort jobs by profit in descending order
    jobs.sort(key=lambda job: job[2], reverse=True)

    # Initialize a result array and a slot array to keep track of free time slots
    result = [None] * m
    slots= [False] * m

    # Iterate over sorted jobs
    for job in jobs:
        job_id, deadline, profit = job

        # Find a free slot for this job (starting from the latest possible slot)
        for free_slot in range(min(m, deadline), -1, -1):
            if slots[free_slot] is False:
                slots[free_slot] = True
                result[free_slot] = job_id
                break
    
    # return the jobs that could be scheduled
    return [job_id from job_id in result if job_id is not None]
```

### Dijkstra's Shortest Path Algorithm
In graph section refer `Single Source Shortest Path`.

### Prim's Minimum Spanning Tree Algorithm
In graph section refer `Minimum Spanning Tree`.

### Kadane's Algorithm
Kadane's algorithm is a famous algorithm used to solve the `Maximum Subarray Problem` in linear time.

#### Problem
The problem is to find the contiguous subarray within a one-dimensional array of numbers - **both positive and negative numbers** - that has the largest sum.

#### Intuition
Kadane's Algorithm makes a greedy choice at each step by selecting the option that maximizes the sum up to the current element. This choice ensures that the local maximum is always the best option for achieving the global maximum.

##### Working:
- The algorithm keeps track of two key pieces of information: the maximum sum of the subarray that ends at the current position and the maximum sum of any subarray encountered so far.
- As you iterate through the array, for each element, you decide whether to include it in the current subarray or to start a new subarray with this element. This decision is based on which option yields a higher sum.
  - If including the element in the current subarray results in a higher sum, you continue with the current subarray.
  - If starting a new subarray with the current element results in a higher sum, you start a new subarray.
- Compare this value with the global maximum and update if needed.

Code
```python
def maximum_subarray_problem(nums):
    max_ending_here = nums[0]
    max_subarray = nums[0]

    for num in nums[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_subarray = max(max_subarray, max_ending_here)
    
    return max_subarray
```

#### Problems based on Kadane's Algorithm
- Maximum Product Subarray

#### Maximum Product Subarray
Given an array of integers, find the contiguous subarray within the array that has the largest product.

##### Intuition
The intuition behind the Maximum Product Subarray is similar to Kadane's Algorithm, but with a twist as the product of two negative numbers is positive, you need to keep track of both the maximum and minimum products upto the current position.

##### Working
- The algorithm keeps track of three key pieces of information:
  - The maximum product of the subarray that ends in the current position
  - The minimum product of the subarray that ends in the current position
  - The maximum product subarray encountered so far.
- As you iterate through the array, for each element: 
  - Swap the current max and current min if the element is negative
  - Update them by considering the current element and their product with the current element.
  - Update the global maximum if needed.

Code
```python
def maximum_product_subarray(nums):
    max_product = min_product = result = nums[0]

    for num in nums[1:]:
        if num < 0:
            # TWIST: negative multiplied by negative is positive
            max_product, min_product = min_product, max_product
        
        # same old same old
        max_product = max(num, max_product * num)
        min_product = max(num, min_product * num)

        result = max(result, max_product)
```
