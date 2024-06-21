from abc import ABC, abstractmethod


class Person(ABC):
    def __init__(self, name: str, yob: int):
        self._name = name
        self._yob = yob

    def get_name(self):
        return self._name

    def get_yob(self):
        return self._yob

    @abstractmethod
    def describe(self):
        pass


class Student(Person):
    def __init__(self, grade: str, name, yob):
        super().__init__(name, yob)
        self.__grade = grade

    def describe(self):
        return print(f"Student - Name: {self.get_name()} - YoB: {self.get_yob()} - Grade: {self.__grade}")


class Teacher(Person):
    def __init__(self, subject: str, name: str, yob: int):
        super().__init__(name, yob)
        self.__subject = subject

    def describe(self):
        return print(f"Teacher - Name: {self.get_name()} - YoB: {self.get_yob()} - Subject: {self.__subject}")


class Doctor(Person):
    def __init__(self, specialist: str, name: str, yob: int):
        super().__init__(name, yob)
        self.__specialist = specialist

    def describe(self):
        return print(f"Doctor - Name: {self.get_name()} - YoB: {self.get_yob()} - Specialist: {self.__specialist}")


class Ward:
    def __init__(self, name: str):
        self.__name = name
        self.__persons: list[Person] = []

    def add_person(self, person: Person):
        self.__persons.append(person)

    def count_doctor(self):
        count = 0
        for i in self.__persons:
            if type(i) is Doctor:
                count += 1

        return print(f"Number of doctors in {self.__name}: {count}")

    def sort_age(self):
        self.__persons.sort(key=lambda x: x.get_yob(), reverse=True)

    def describe(self):
        for person in self.__persons:
            person.describe()

    def compute_average(self):
        avg_yob = 0
        numb_tech = 0
        for person in self.__persons:
            if type(person) is Teacher:
                numb_tech += 1
                avg_yob += person.get_yob()

        if numb_tech == 0:
            return print(f"There is no teacher in {self.__name}")

        return avg_yob / numb_tech
