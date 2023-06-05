def compare(a, b):
    return a <= b


def compare_string(a: str, b: str):
    return a <= b

def compare_string_mine(a:str,b:str):
    left = 0
    while left < len(a) and left < len(b):
        if a[left] < b[left]:
            return True
        elif a[left]==b[left]:
            left+=1
        else:
            return False

        return len(a)-left <= len(b)-left


def mergeSort(arr):
    if len(arr) > 1:

        
        mid = len(arr) // 2

        
        L = arr[:mid]

        
        R = arr[mid:]

        
        mergeSort(L)

        
        mergeSort(R)

        i = j = k = 0

        
        while i < len(L) and j < len(R):
            if compare_string_mine(L[i], R[j]):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1





def printList(arr):
    for i in range(len(arr)):
        print(arr[i], end=" ")
    print()




















if __name__ == '__main__':
    
    
    
    
    
    

    my_str = "Hello this Is an Example With cased letters"

    
    

    
    words = [word.lower() for word in my_str.split()]
    print(words)
    mergeSort(words)
    print(words)



