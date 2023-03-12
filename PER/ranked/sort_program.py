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

        # Finding the mid of the array
        mid = len(arr) // 2

        # Dividing the array elements
        L = arr[:mid]

        # into 2 halves
        R = arr[mid:]

        # Sorting the first half
        mergeSort(L)

        # Sorting the second half
        mergeSort(R)

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if compare_string_mine(L[i], R[j]):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


# Code to print the list


def printList(arr):
    for i in range(len(arr)):
        print(arr[i], end=" ")
    print()

# Program to sort alphabetically the words form a string provided by the user

# my_str = "Hello this Is an Example With cased letters"
#
# # To take input from the user
# #my_str = input("Enter a string: ")
#
# # breakdown the string into a list of words
# words = [word.lower() for word in my_str.split()]
#
# # sort the list
# words.sort()
#
# # display the sorted words
#
# print("The sorted words are:")
# for word in words:
#    print(word)
# Driver Code
if __name__ == '__main__':
    # arr = [12, 11, 13, 5, 6, 7]
    # print("Given array is", end="\n")
    # printList(arr)
    # mergeSort(arr)
    # print("Sorted array is: ", end="\n")
    # printList(arr)

    my_str = "Hello this Is an Example With cased letters"

    # To take input from the user
    # my_str = input("Enter a string: ")

    # breakdown the string into a list of words
    words = [word.lower() for word in my_str.split()]
    print(words)
    mergeSort(words)
    print(words)


# This code is contributed by Mayank Khanna
