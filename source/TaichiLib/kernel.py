from .common import *

@ti.kernel
def add(self:ti.template(),_:ti.template()):
    for i in ti.grouped(self):
        self[i]+=_[i]
