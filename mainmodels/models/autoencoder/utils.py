# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/5/22

from __future__ import absolute_import
from __future__ import print_function

from enum import Enum, unique

@unique
class InputType(Enum):
    """Enum to specify the data type requested"""
    validation = 'validation'
    train = 'train'
    test = 'test'

    def __str__(self):
        """Return the string representation of the enum"""
        return self.value

    @staticmethod
    def check(input_type):
        """Check if input_type is an element of this Enum"""
        if not isinstance(input_type, InputType):
            raise ValueError("Invalid input_type, required a valid InputType")