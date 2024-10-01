from .common import *

@ti.kernel
def add(self:ti.template(),_:ti.template()):
    for i in ti.grouped(self):
        self[i]+=_[i]

@ti.kernel
def add_scaled(self:ti.template(),_:ti.template(),scale:float):
    for i in ti.grouped(self):
        self[i]+=_[i]*scale
