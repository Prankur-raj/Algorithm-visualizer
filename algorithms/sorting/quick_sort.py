def quick_sort_steps(arr):
    steps = []
    steps.append(arr.copy())  # show original array first
    _quick_sort(arr, steps)
    return steps

def _quick_sort(arr, steps):
    if len(arr) <= 1:
        return arr
    pivot = arr[-1]
    left = [x for x in arr[:-1] if x <= pivot]
    right = [x for x in arr[:-1] if x > pivot]
    left = _quick_sort(left, steps)
    right = _quick_sort(right, steps)
    sorted_arr = left + [pivot] + right
    steps.append(sorted_arr.copy())
    return sorted_arr