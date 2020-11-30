# -*- coding:utf-8 -*-
# @Time: 2020/11/30 7:18 下午
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: test.py

class A:
    def __init__(self, a):
        self.a = a
        print("a = %d",a)
        print("aaaaaaaa")

    def fun(self):
        print("afun")

class B(A):
    def __init__(self, a,b):
        super().__init__(a)
        self.b = b
        print("b=%d",b)
        print("bbbbbbb")

    def fun(self):
        print("bfun")

b = B(2,3)
b.fun()
