# Copyright 2019 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# =============================================================================
from enum import Enum

import six


class WarningAction(Enum):
    IGNORE = 'ignore'
    SAVE = 'save'

    # we may eventually want to support logging and raising python Warnings
    # LOG = 'log'
    # RAISE = 'raise'


IGNORE = WarningAction.IGNORE
SAVE = WarningAction.SAVE
# LOG = WarningAction.LOG
# RAISE = WarningAction.RAISE


class WarningHandler(object):
    def __init__(self, action=None):
        self.saved = []

        if action is None:
            action = WarningAction.IGNORE
        elif isinstance(action, WarningAction):
            pass
        elif isinstance(action, six.string_types):
            action = WarningAction[action.upper()]
        else:
            raise TypeError('unknown warning action provided')

        self.action = action

    # todo: let user override __init__ parameters with kwargs
    def issue(self, msg, func=None):
        """Issue a warning.

        Args:
            msg (str): The warning message
            func (function):
                A function that is executed in the case that the warning level
                is not IGNORE. The warning is issued only if the function
                returns True. This is used to lazily execute expensive tests.

        """

        if self.action is IGNORE:
            return
        elif (func is not None and not func()):
            # func condition was not met so do nothing
            return
        elif self.action is SAVE:
            self.saved.append(msg)
        else:
            raise TypeError("unknown action")

    # some hard-coded warnings for convenience

    def chain_length(self, embedding, length=7):
        def func():
            return any(len(chain) > length for chain in embedding.values())

        self.issue('chain length greater than {}'.format(length), func)
