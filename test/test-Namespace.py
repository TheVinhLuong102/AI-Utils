from __future__ import print_function

from arimo.util import Namespace


n = Namespace(a=1, b=Namespace(c=2))

print(n)
print(len(n))
print(n.keys())
print(n.keys(all_nested=True))
