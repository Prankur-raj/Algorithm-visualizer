def merge_sort_steps(arr):
    steps = []
    steps.append(arr.copy())  # ← show original array first
    _merge_sort(arr, steps)
    return steps

def _merge_sort(arr, steps):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = _merge_sort(arr[:mid], steps)
    right = _merge_sort(arr[mid:], steps)
    sorted_arr = merge(left, right)
    steps.append(sorted_arr.copy())  # ← snapshot after each merge
    return sorted_arr

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result