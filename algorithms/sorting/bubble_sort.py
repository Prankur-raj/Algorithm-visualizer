def bubble_sort_steps(arr):
    arr=arr.copy()
    steps=[]
    steps.append(arr.copy())


    for i in range(len(arr)):
        for j in range (len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                steps.append(arr.copy())
    return steps