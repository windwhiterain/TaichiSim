from typing import Any, Callable, Generator, Iterable
from pathlib import Path
import taichi as ti 

class FieldIterator:
    def __init__(self,field:ti.Field) -> None:
        assert(len(field.shape)==1)
        self.field=field
        self.idx=0
    def __next__(self):
        if self.idx<self.field.shape[0]:
            ret=self.field[self.idx]
            self.idx+=1
            return ret
        else:
            raise StopIteration
    def __iter__(self):
        return self
        
class FieldIterable:
    def __init__(self,field:ti.Field) -> None:
        assert(len(field.shape)==1)
        self.field=field
    def __iter__(self)->FieldIterator:
        return FieldIterator(self.field)
    def __len__(self)->int:
        return self.field.shape[0]
    

class GeneratorIterable:
    def __init__(self,generatable:Callable[[],Generator]) -> None:
        self.generatable=generatable
    def __iter__(self):
        return self.generatable()

class Linq:
    def __init__(self,iter:Iterable|ti.Field) -> None:
        if isinstance(iter,ti.Field):
            self.iter=FieldIterable(iter)
        else:
            self.iter=iter
    def filter_map(self,function:Callable[[int,Any],tuple[bool,Any]]):
        def ret():
            for k,v in enumerate(self.iter):
                f,m=function(k,v)
                if f:
                    yield m
        return Linq(GeneratorIterable(ret))
    def map(self,function:Callable[[int,Any],Any]):
        return Linq(function(k,v) for k,v in enumerate(self.iter))
    def filter(self,function:Callable[[int,Any],Any]):
        return Linq(filter(lambda x:function(x[0],x[1]),enumerate(self.iter)))
    def __str__(self) -> str:
        ret=""
        ret+="["
        for i in self.iter:
            ret+=i.__repr__()
            ret+=","
        ret+="]"
        return ret
    def log(self,path:str):
        with open(Path(__file__).parent/path,'w') as f:
            f.write("[")
            for i in self.iter:
                f.write(i.__repr__())
                f.write(",")
            f.write("]")