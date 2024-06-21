class MyQueue:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__data = []

    def is_empty(self):
        return len(self.__data) == 0

    def is_full(self):
        return len(self.__data) == self.__capacity

    def dequeue(self):
        if self.is_empty():
            print("Queue has no element")
            return None
        else:
            return self.__data.pop(0)

    def enqueue(self, value):
        if self.is_full():
            print("Queue has full")
        else:
            self.__data.append(value)

    def front(self):
        if self.is_empty():
            print("Queue has no element")
            return None
        else:
            return self.__data[0]
