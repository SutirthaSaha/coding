# Binary Search
Whenever in the input you see a sorted array or nearly sorted array. Think of **binary search**.

## Implementation
Here is a basic implementation of binary search.
```python
def binary_search(nums, val):
    n = len(nums)
    start, end = 0, n-1

    while start <= end:
        mid = start + (end-start) // 2
        if nums[mid] == val:
            return mid
        elif nums[mid] > val:
            end = mid - 1
        else:
            start = mid + 1
    
    return -1
```
This is an example of divide and conquer as we divide the input into half into one we need and one we don't. On dividing the search space into half everytime, we get a complexity of `O(n)`.

## Identification
- The array is sorted

## Problem
### Descending sorted array
If the input array provided is sorted in a descending order, the only modification in the solution would be the condition change.
In the original problem if the array element was greater than the value to search, we used to go to the first half, **but now it would be in the second half** as the array is sorted in descending order, the lower value would be in the second half.

```python
def binary_search_desc(nums, val):
    n = len(nums)
    start, end = 0, n-1

    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] == val:
            return mid
        elif nums[mid] > val:
            start = mid + 1 # here is where the condition changes
        else:
            end = mid - 1
    
    return -1
```

### Order not known Search
Given an array which is sorted but the manner of sorting is not provided.
Solution:
- The modification would be that we need to identify the order of sorting first. Take the first and the last element of the array, if the first element is lesser than the last element - **sorted in ascending order** otherwise in descending order.
- After identification call the relevant binary search as we discussed earlier.

### First and last occurence of an element
Given an input array find the first and the last occurence of an element in the array.
The modification would be that we won't return after finding the element in the array.
We would store it as a possible solution and then:
- for the first occurence - continue search in the left search space
- for the last occurence - continue search in the right search space

```python
def first_occurence(nums, val):
    n = len(nums)
    start, end = 0, n-1
    result = -1
    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] == val:
            result = mid # this is where we select as a possible solution
            end = end - 1 # then move to the left subarray 
        elif nums[mid] > val:
            end = mid - 1
        else:
            start = mid + 1
    return result
```

The last occurence would be almost the same, but instead of moving to the left -we would move to the right search space.
```python
if nums[mid] == val:
    result = mid
    start = mid + 1 # move to the right search
```

### Count of elements in the sorted array
Given a sorted array of elements, find the number of occurences of the element.
We can use the above concept in finding out the count:
- Find the **first occurence** of the element
- Find the **last occurence** of the element
- Count of elements = (last - first) + 1

### How Many Times a Sorted Array Has Been Rotated
Given a sorted array which is rotated an unknown number of times, find the number of times the array has been rotated.  
  
The intuition behind solving this problem lies in finding the minimum element of this sorted array. The index of this minimum element would be the number of times the array has been rotated.  
  
While this can be found using linear search (`O(n)` complexity), we want to solve it in `O(log n)` using a method analogous to binary search.  
  
#### Two Factors of Binary Search:  
  
1. **Condition of Matching**  
  - The minimum element is the one which is smaller than both its adjacent elements.  
    - Left neighbor lies in: `(index + n - 1) % n`  
    - Right neighbor lies in: `(index + 1) % n`  
  
2. **How to Move/Divide Search Space**  
  - Always move to the unsorted part of the array, as the minimum always lies in the unsorted part.  
  - Conditions:  
    - If the mid element is lesser than the first element, the minimum lies in the first half.  
    - If the mid element is greater than the last element, the minimum lies in the second half.  

Code  
```python  
def find_rotation_count(arr):  
    n = len(arr)  
    start, end = 0, n - 1  
  
    while start <= end: 
      if arr[start] <= arr[end]:  # Case when the subarray is already sorted  
        return start
        
      mid = (start + end) // 2
      next_index = (mid + 1) % n
      prev_index = (mid + n - 1) % n 

      # Check if mid element is the minimum  
      if arr[mid] <= arr[next_index] and arr[mid] <= arr[prev_index]:  
        return mid
        
      # Decide whether we need to move to the left half or the right half  
      if arr[mid] <= arr[end]:  
        end = mid - 1  # unsorted part on the left
      else:  
        start = mid + 1  # unsorted part on the right
```

### Find an element in a sorted rotated array
Given a sorted array which is rotated an unknown number of times, find whether an element exists or not.

**Focus on what you already know, and try to build on top of it**

#### Intuition
- We know how to find out the minimum in the rotated sorted array.
- If we find out the index of the minimum element in the array, can we do something with it?
- We can divide the array into 2 sorted search spaces: (0 to min_index - 1) and (min_index to last)
- Perform binary search in both these subarrays and return if exists in either of these.

### Searching in a nearly sorted array
Given an input array which is nearly sorted and the index of the element can be either (i-1, i or i + 1) if i is the index if was sorted completely.

On comparing with the binary search we already know, we need to find things:
- **Condition of matching**: The element can be present in - (mid-1, mid, mid+1) - either of these positions.
  - To ensure that `mid-1` is in search space - check `mid - 1` >= `start`.
  - Same, ensure `mid + 1` <= `end`.
- **How to move/divide search space**:
  - If arr[mid] > x, it means the target element must be in the left sub-array. Since we've already checked mid and mid+1, we move to **end = mid - 2**.
  - If arr[mid] < x, it means the target element must be in the right sub-array. Since we've already checked mid and mid-1, we move to **start = mid + 2**.

```python
def binary_search_nearly_sorted(arr, x):  
    start, end = 0, len(arr) - 1  
      
    while start <= end:  
        mid = (start + end) // 2  
          
        # Check if the middle element is the target  
        if arr[mid] == x:  
            return mid  
          
        # Check if the element is present at mid-1  
        if mid - 1 >= start and arr[mid - 1] == x:  
            return mid - 1  
          
        # Check if the element is present at mid+1  
        if mid + 1 <= end and arr[mid + 1] == x:  
            return mid + 1  
          
        # Move to the left half if element is smaller than mid element  
        if arr[mid] > x:  
            end = mid - 2  
        else:  
            # Move to the right half if element is greater than mid element  
            start = mid + 2  
      
    # Element is not present in array  
    return -1
```

### Finding floor of an element in a sorted array
Given an sorted inout array, find the floor on an element in the array.
If the element already exists in the array, it would be the floor itself.

#### Intuition
- if the number exists - return the number itself
- if the mid value is less than the number - it is a possible result and continue with the right subarray - to find another element lesser as well as closer to the provided value.
- otherwise continue with the left subarray.

Code
```python
def find_floor(nums, val):
    result = -1
    n = len(nums)
    start, end = 0, n-1

    while start <= end:
        mid = start + (end-start) //2

        if nums[mid] == val:
            return nums[mid]
        elif nums[mid] < val:
            result = mid
            start = mid + 1
        else:
            end = mid - 1

    if result != -1:
        return nums[result]
    return result
```

### Find ceil of an element in a sorted array
Given an sorted inout array, find the ceil on an element in the array.
If the element already exists in the array, it would be the ceil itself.

#### Intuition:
- The problem is exactly the same as the previous, only here the logic would be reverse.
- If the number exists - return the number itself
- If the mid value is greater than the number - it is a possible result and continue with the left subarray - to find another element greater as well as closer to the provided value.
- Otherwise continue with the right subarray.

### Next letter problem
Given a sorted array of alphabets, find the next lexigraphical letter for a provided letter in the array.
Even if the provided letter exists, we need to provide the next letter.

#### Intuition
- Same as the previous problem of finding the ceiling of a number in a sorted array. In this case the elements are alphabets.
- If the mid value is greater than the alphabet - it is a possible result and continue with the left subarray.
- Otherwise continue with the right subarray.

### Find position of an element in an infinite sorted array

### Index of first 1 in a binary sorted infinite array

### Minimum difference element in a sorted array

## Binary Search on Answer
Earlier the pre-requisite for binary search woud be sorted array.
But there are situations where binary search can be used for un-sorted array as well - **binary search on answer**.
We would have to follow the 2 steps for binary search in order to find the answer:
- Condition of matching
- How to move/divide search space
 
## Problems
### Peak Element
Given an unsorted array, find the index of the peak element.
**Peak element**: the number is greater than both its adjacent elements. There can be multiple peak elements - local peak, global peak - can return any one of them.

#### Intuition
- **Condition for matching**: The element mush be greater than both its adjacent elements. For the last element consider only the left element and same for the first element.
- **How to move/decide search space**: Whichever side has greater element we can move to that side. If both sides have greater element we can go on either sides.

Code:
```python
def peak_element(nums):
    n = len(nums)
    start, end = 0, n-1

    while start <= end:
        mid = start + (end-start) // 2
        if (mid == 0 or arr[mid] > arr[mid-1]) and (mid == n-1 or arr[mid] > arr[mid+1]):
            return mid
        elif mid > 0 and arr[mid] < arr[mid-1]:
            end = mid - 1
        else:
            start = mid + 1
    return -1
```

### Find maximum in a bitonic array
Given a bitonic array, find the maximum element in the array.
**Bitonic Array**: A bitonic array is an array of integers which has the following properties:
- The array initially increases and then decreases.
- There is exactly one peak element which is the maximum element in the array.

Consider the following bitonic array: [1, 3, 8, 12, 4, 2]  

In this example:
- The array increases from 1 to 12.
- After reaching the peak 12, it decreases to 2.
- The maximum element in the array is 12.

#### Intuition
This problem would be same as the previous one where we find the peak element in the array. Since in this array, there would be only one peak which is the **maximum**, we just need to provide the same code and everything would work.

### Find an element in the bitonic array
Given a bitonic array, find whether a particular element exists in the array.

#### Intuition
- We already know how to find the maximum in a bitonic array.
- Observation: Till the maximum element index, the array is increasing. Post that it is decreasing.
- So we can divide the array into 2 halves - ascending and descending sorted array and perform binary search on both of them for finding the element. This is something we have already seen earlier.

### Search in a row-wise and column-wise sorted matrix
Given a row-wise and column-wise sorted matrix, find an element in it.

#### Intuition
- First we need to identify the row in which the element would lie.
- Then we need to find out in the sorted row if the element would exist or not - this would be a normal binary search.

**Finding the row**:
- **Condition for matching**: If the element is greater than or equal to the first element of the row and is lesser than or equal to the last element.
- **How to move/decide search space**: This can be done using the first element of the row in question. If the element is lesser than the first element - `end = mid - 1`.

The rest of the problem remains the same. Time complexity: `O(log m + log n)`.

#### Another approach
- start with top right element - `(i, j)`.
- if element is equal - return
- if element is greater than search value - `(i. j-1)`. Ignore the `j th` column as it would have values greater than `(i, j)`.
- else (when the element is lesser than the search value): `(i+1, j)`. The `i th` column to the left contains elements only lesser than current.

Although the time complexity of this solution is `O(m + n)`, this is just another approach to help think in a different way.

### Allocate Pages of Books
Given an array where each element denotes the number of pages in the book at that index. There are `m` number of students, and the task is to allocate books to students such that:
- Each student gets at least one book.
- Each book can be allocated to only one student.
- The maximum number of pages assigned to a student is minimized.
Find the minimum value of the maximum number of pages assigned to a student.

**Example**:
Consider the following example:
- arr = [12, 34, 67, 90]  
- m = 2

The optimal allocation is:
- Student 1 gets books with pages [12, 34, 67] (total 113 pages)
- Student 2 gets books with pages [90] (total 90 pages)

The minimum value of the maximum number of pages assigned to a student is 113.

### Intuition
1. **Determine the Range**:
   - The minimum possible number of pages that any student can read is the maximum number of pages in a single book. This is because at least one student will have to read the largest book.
   - The maximum possible number of pages that any student can read is the sum of all pages in the books. This scenario assumes that one student reads all the books.
2. **Binary Search Approach**:
   - Use binary search to explore potential solutions. The idea is to find the minimum possible value of the maximum number of pages that can be assigned to a student.
3. **Check Feasibility**:
   - For a given mid-point (let's call it `middle value`), determine whether it is possible to allocate books such that no student reads more than `middle value` pages.
   - **If it is possible**: This means that `middle value` is a valid solution, but there might be a smaller valid solution. Therefore, store this `middle value` as a potential result and continue searching in the lower half.
   - **If it is not possible**: This means that `middle value` is too small, so we need to search in the upper half.
4. **Iterate Until Convergence**:
   - Repeat the process until the binary search bounds converge. The goal is to find the minimum value that allows all books to be allocated to `m` students under the given constraints.
   - By iteratively adjusting the bounds and checking feasibility, we can efficiently determine the optimal allocation of books."

```python
def allocate_books(books, m):
    n = len(books)
    start, end = max(books), sum(books)
    result = -1

    def is_sufficient(max_pages):
        count = 1
        curr_page_count = 0
        for pages in books:
            if curr_page_count + pages > max_pages:
                count = count + 1
                if count > m:
                    return False
                curr_page_count = 0                         
            curr_page_count = curr_page_count + pages
        return True
    
    while start <= end:
        mid = start + (end - start) // 2
        if is_sufficient(mid):
            result = mid
            end = mid - 1
        else:
            start = mid + 1
    
    return result
```
