#! /usr/bin/env python3
#-*- coding:utf-8 -*-

'''
Defines Singleton Class. Singleton Pattern is a simple disgining mode. Any class inherited from `Singleton` will only exists one implement of this class.

View [here](https://www.runoob.com/design-pattern/singleton-pattern.html) for more information about Singleton Parttern, and [here](https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa) for the reference of creating a singleton in Python.

'''

class Singleton(object):
    ''' This class and it's derived class will only have one implement. '''
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not class_._instance:
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

